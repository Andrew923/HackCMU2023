CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230209000000_e20230209235959_p20230210021527_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-10T02:15:27.903Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-09T00:00:00.000Z   time_coverage_end         2023-02-09T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxc7��  T          @�33���H���\���ͿW
=Cb^����H��ff�p���=qC^��                                    Bxc7�f  "          @�=q���H��ff?�  A33C[�����H�����Q��  C[�                                     Bxc7�  T          @��
�|(����?E�@���Cic��|(���������y�Ch�                                    Bxc7��  "          @�zῺ�H��=q?��
A%�C��ÿ��H��(��33��ffC��=                                    Bxc7�X  T          @�R�E���\)?��@��Cs���E������=q���Cq�)                                    Bxc7��  T          @�{�#�
�ۅ?O\)@�G�Cy���#�
��{��R���CxO\                                    Bxc7�  �          @�����Ϯ?�(�A�
Cm@ ����녿�Q��fffCl�                                    Bxc7�J  
�          @���(���(�?��
@���Cjz���(���������mG�Ciu�                                    Bxc8�  �          @�p���\)�ƸR?��
A7�Ck��\)�ƸR����9��Ck�q                                    Bxc8�  T          @�ff��ff���H?��A_�CjJ=��ff��  �������Ck�                                    Bxc8<  �          @��
������{?˅AH  Ck��������  �����)p�Ck�R                                    Bxc8-�  �          @�=q��  ���R?�z�A333Ci\)��  ���R��z��333Ci\)                                    Bxc8<�  T          @�{������G�?�{A(��CeE�������ÿ�\)�*{Ce@                                     Bxc8K.  T          @�33���
��=q?�z�A1p�Cc�����
���
��(���Cc�\                                    Bxc8Y�  T          @������?��A)��Cbh������R��(���RCb��                                    Bxc8hz  T          @陚���\��p�?O\)@�z�Ca� ���\���R��
=�Up�C`G�                                    Bxc8w   T          @���  ��(�>���@��C^�3��  ���R�   ���C\ff                                    Bxc8��  �          @ᙚ������  ��=q�
�HCV�������Z�H���(�CR��                                    Bxc8�l  	�          @�  ��{��
=��G��&�\C\�q��{�XQ��L(���{CU�                                    Bxc8�  T          @�Q�������\)��Q��5Cl
������(��E���HCg�                                    Bxc8��  "          @�\�u��ƸR��Q�+�CnY��u����R�<�����Cj��                                    Bxc8�^  
�          @�(���ff���ÿ@  ���HCdQ���ff���\�G����C^��                                    Bxc8�  �          @�z���(����\�����P��Cgٚ��(���Q��<(��îCcY�                                    Bxc8ݪ  "          @�p��e�������G�Co���e���
=�W���{Ck�                                    Bxc8�P  
�          @�(��\)����Y���ٙ�Ck�\)�����]p���{CfJ=                                    Bxc8��  T          @ᙚ��  ��녿�����Ce�R��  ��ff�[���33C_                                    Bxc9	�  �          @�p�������\)�aG���G�Cfs3������ff�S�
��ffC`��                                    Bxc9B  
�          @���p����
�Tz����Cd����p����
�N{�ՙ�C_)                                    Bxc9&�  "          @���ff���׿\(���{Cg���ff��  �Tz���\)CaJ=                                    Bxc95�  
�          @��H�g�����<�>W
=Co
=�g�����0����
=Ck��                                    Bxc9D4  
�          @��HQ��˅>W
=?�Q�Cs�=�HQ�����0  ����CqaH                                    Bxc9R�  
�          @����q���ff���s�
Cn�f�q������R�\��33Cjh�                                    Bxc9a�  
(          @޸R�h����p����}p�Cn^��h�������J=q�י�Cj\                                    Bxc9p&  �          @���`  �����.{��{Co�3�`  ��G��<(�����Cl��                                    Bxc9~�  T          @����XQ�����W
=��Q�Cq@ �XQ����
�AG����
Cm��                                    Bxc9�r  "          @���^{���H���uCpT{�^{��33�:=q��=qCm
=                                    Bxc9�  
�          @�  ��R��z���\��=qCz���R��������2G�Cs�                                    Bxc9��  "          @�(��HQ���������z�RCq
=�HQ���{��
=�z�Ci5�                                    Bxc9�d  �          @���Z�H����xQ��Q�Co8R�Z�H�����`  ��{Ci��                                    Bxc9�
  
�          @�{�L����=q?&ff@��
Cr33�L����{�{��=qCp�H                                    Bxc9ְ  "          @�ff�P����  ?n{@�\)Cq}q�P����Q��Q���z�Cpz�                                    Bxc9�V  
Z          @ۅ�]p���{?�  A*�\Cn���]p���(���  �L��Cnk�                                    Bxc9��  	�          @�z���{���������Ch\)��{��=q�>�R��G�Cc�                                    Bxc:�  T          @�G��|(���
=��
=�e�Ci  �|(��w��x����\C`��                                    Bxc:H  �          @�
=��������Q��k�Cb�\����W
=�hQ���CYW
                                    Bxc:�  
(          @�G��x����녿n{� z�CiǮ�x����G��P������Cc�{                                    Bxc:.�  
�          @����`�����Ϳ�p��"�\Co:��`�����s33��Ci)                                    Bxc:=:  �          @�ff��
=���ÿ.{��ffCi�{��
=��=q�P  ��G�Cd�=                                    Bxc:K�  "          @�  �������Ǯ�EChu�������>�R��Cd&f                                    Bxc:Z�  �          @���33��>#�
?�G�Ch�{��33���
�p�����Ce��                                    Bxc:i,  �          @�G������녾����{Cis3�����=q�9�����HCe�
                                    Bxc:w�  
�          @�33����\)>�?s33Cg�R����(��'����Cd��                                    Bxc:�x  �          @����r�\��p��L�;ǮCm\)�r�\����0  ��  Cj!H                                    Bxc:�  �          @�Q��dz����H��\)�   Cq���dz����\�Dz���33Cn��                                    Bxc:��  
�          @��\�y����G��L�;ǮCp
�y�������H����p�Cm�                                    Bxc:�j  "          @�=q���
��zᾅ���Cn8R���
��G��P����G�Cj��                                    Bxc:�  
�          @������\���
�   �k�CnY����\��p��]p�����Cj:�                                    Bxc:϶  
�          @����z��Ӆ�.{��(�Cm���z�����J=q��
=Cj��                                    Bxc:�\  �          @����p���  >W
=?��CmT{��p���(��1G���Q�Cj�                                    Bxc:�  �          @�Q��o\)��z�>8Q�?�\)Co���o\)�����0  ��Q�Cm�                                    Bxc:��  
�          @������=q��\)�r=qC|Y����  �����"ffCwT{                                    Bxc;
N  �          @�
=�)����{��\)�o�
Cw�)�)����z����H�
=Cq�                                    Bxc;�  �          @���<���θR��z��4Q�Cup��<�������{��Cp                                      Bxc;'�  "          @�33�z=q���(���\)Cm���z=q��\)�U����Ci8R                                    Bxc;6@  T          @�33�`����(��n{��Q�Cq.�`�������n{��ClO\                                    Bxc;D�  
�          @�=q��G����H>�ff@i��C_ٚ��G����׿���mC]�H                                    Bxc;S�  �          @�{��{����?E�@�(�C]{��{��(���{�5p�C\.                                    Bxc;b2  T          @�{������33?c�
@�z�Cg5��������У��Y��CfB�                                    Bxc;p�  �          @�p��x����(�?J=q@��CkT{�x����(����xQ�Cj
                                    Bxc;~  �          @�ff�g
=��?&ff@�p�Co�R�g
=�����p����Cn#�                                    Bxc;�$  
�          @�{�
=��(�>�?�  C}���
=��p��Fff��\)C{��                                    Bxc;��  T          @�G��h����(�>���@L(�CoQ��h�������=q���Cm:�                                    Bxc;�p  �          @陚�!��ڏ\>��
@!�Cy�f�!��Ǯ�2�\��(�Cw��                                    Bxc;�  �          @�Q����
=������C����ƸR�n{��z�C|��                                    Bxc;ȼ  T          @�ff�_\)��33���Ϳ8Q�Cr�q�_\)��=q�J�H���HCp�                                    Bxc;�b  �          @��dz���Q�>�  ?��HCq@ �dz����,������Cn�                                    Bxc;�  �          @陚�dz��ʏ\����Mp�Cp���dz���
=�Mp���\)Cl�)                                    Bxc;��  �          @��_\)��p�@XQ�A�\)Cg�R�_\)��
=?��
A  Cmp�                                    Bxc<T  T          @����r�\�i��@���B)�C_�)�r�\���@)��A���Cj��                                    Bxc<�  
Z          @�(��{���\)>���@G�Cm� �{���  �(���Q�Ck�H                                    Bxc< �  
Z          @�p��`����  ?ǮAI�Co�3�`����=q��ff�((�Co�3                                    Bxc</F  "          @��W���
=@4z�A�
=Cp���W���33�����Cr�R                                    Bxc<=�  
�          @�R�Vff���@s�
A���Cl�Vff�ȣ�?�\)A
=Cq��                                    Bxc<L�  "          @��H�P����33@\)B	33Cl{�P����(�?�33A6{Cq�3                                    Bxc<[8  T          @�G��A����@�Q�B,CiG��A���Q�@(�A�ffCrB�                                    Bxc<i�  �          @�\�AG����@�33B"�RCk�{�AG���  @��A��CsB�                                    Bxc<x�  "          @���K���\)@�G�B	ffCmp��K���Q�?��A3\)Cs\                                    Bxc<�*  
(          @�\)�<(���G�@\)Bp�Cp�q�<(��У�?��HAG�Cu�                                     Bxc<��  
�          @����@������@w
=A��Cp�{�@���љ�?�ffA  CuJ=                                    Bxc<�v  	�          @����QG����H@j�HA�=qCp��QG��ۅ?0��@�{Ct��                                    Bxc<�  "          @�Q��\(����@��
B 33Cnz��\(��ۅ?���A��CsY�                                    Bxc<��  
�          @�  �K����H@��\B��CpaH�K���?�33A&=qCuaH                                    Bxc<�h  �          @�  �Fff���@��B	�Cp޸�Fff��p�?�Q�A+�
Cu�H                                    Bxc<�  	�          @��Tz����H@���B0�Ch���Tz��Ǯ@4z�A�(�Cr
=                                    Bxc<��  
�          @�\�>{���@���B5��Cl��>{��33@8��A��Ct�f                                    Bxc<�Z  "          @�z��8�����H@��B1�\Cm�q�8���θR@.�RA��\Cu޸                                    Bxc=   �          @����^{����@0  A�
=Co)�^{��p����ͿL��Cq��                                    Bxc=�  
Z          A   �~{��{��G��G�Cp@ �~{���J�H���RCmJ=                                    Bxc=(L  �          @�
=�vff��p��O\)���Cp�H�vff��33�q�����Cl�H                                    Bxc=6�  "          @��H�u�׮�n{��33CpT{�u��(��s�
�陚Ck��                                    Bxc=E�  T          @�(��vff��\)��\)�
=qCoB��vff��{�G
=����Ck��                                    Bxc=T>  T          @����W
=��z�?�\Aj=qCpG��W
=��=q��G��  Cq�                                    Bxc=b�  �          @�z�^�R��(�@�(�B=C��Ϳ^�R��p�@$z�A��C�"�                                    Bxc=q�  
�          @���N{��G�@HQ�A�{Co���N{��33>�p�@?\)Cs{                                    Bxc=�0  �          @���\(���Q�@/\)A���Cn
=�\(�����=#�
>�z�Cp�\                                    Bxc=��  T          @���HQ���33@`��A�ffCm33�HQ�����?��\A\)Cr{                                    Bxc=�|  
�          @���*=q��  @FffAЏ\Cu!H�*=q�У�>��@��Cw�=                                    Bxc=�"  T          @�{��  ���@j=qA��C|����  ��z�?Tz�@�p�C@                                     Bxc=��  �          @�{�Y����@��B,ffC�S3�Y�����@	��A��C�n                                    Bxc=�n  �          @�R�Z=q��Q�?fff@���Cqk��Z=q���׿�
=�y�CpxR                                    Bxc=�  T          @�������\)@�{B�RC�{�������
?���AM��C�p�                                    Bxc=�  
�          @��Ǯ��(�@��B�RC}�\�Ǯ�ڏ\?�G�AaG�C��\                                    Bxc=�`  "          @�  �����@�{BQ�C����߮?��\A!G�C�B�                                    Bxc>  T          @��Ǯ��33@�\)B,33C}��Ǯ��Q�@��A�ffC��                                    Bxc>�  
�          @�  ���
����@�ffBP��CyǮ���
��
=@U�Aۙ�C�R                                    Bxc>!R  T          @���U�����?�AZ=qCq(��U���p���\)�p�Cq��                                    Bxc>/�  
�          @��Z=q��G�?�AmCp�
�Z=q�Ǯ�xQ���  Cq^�                                    Bxc>>�  
�          @��H�Dz���Q�?��A{Cs޸�Dz��Å�޸R�c�
CsQ�                                    Bxc>MD  "          @��
�6ff��Q�>�z�@
=CvaH�6ff����#�
���RCt�
                                    Bxc>[�  T          @���\)�z�H@���B
=C`� �\)����@\)A�z�CjJ=                                    Bxc>j�  "          @�R�������@���B��C`��������R?�p�AW�ChY�                                    Bxc>y6  T          @���Z�H����@��B!�HCi��Z�H��ff@=qA�ffCq&f                                    Bxc>��  �          @�{��  ���@�
=Bz�Cd��  ����@�A�Q�Cl�R                                    Bxc>��  T          @�������Q�@���B�CcT{�����z�@��A��\Ck��                                    Bxc>�(  "          @�=q��p�����@��B�CcQ���p����
@�A��Ck�                                     Bxc>��  
�          @����p���(�@5�A�G�C_\)��p���(�>�ff@VffCcff                                    Bxc>�t  �          @��H������Q�@o\)A���Cbh��������?�=qA!��Ch�                                     Bxc>�  
�          @�(���{����@aG�A�z�Can��{��G�?��\@�
=Cf�R                                    Bxc>��  T          A����ff��G�@=p�A�G�Ci:���ff�޸R���
�.{Ck�                                    Bxc>�f  
�          A33��=q���
@+�A��Cg�q��=q��p��L�Ϳ�Cj#�                                    Bxc>�  
�          A ��������@'
=A��Cg33������\)�8Q쿢�\Ci�f                                    Bxc?�  �          A��������
=@'�A�
=CgQ������У׾B�\����Ci��                                    Bxc?X  
�          @����G���\)@1�A���Cf!H��G����
=�\)>��HCi{                                    Bxc?(�  
�          @��R��p�����@,(�A�=qCd�3��p���Q�<�>8Q�Cg�{                                    Bxc?7�  
�          @����
=��{@�Az{Cg�\��
=���ÿ(����RCi
                                    Bxc?FJ  �          @��
�����(�?�Q�Ad��Cfs3������Ϳ=p���Cg�3                                    Bxc?T�  �          @����=q���
?��RAj�\Cf�{��=q����333��33Cg�                                    Bxc?c�  
�          @��\�����p�=�G�?Tz�Ce5��������=q��Cb��                                    Bxc?r<  �          @�33������\�5��ffC^�=��������333��33CZ\                                    Bxc?��  �          @�\���
��  >�?�G�CZ�R���
�������hQ�CX�{                                    Bxc?��  T          @�z���33����>�(�@L��C^����33��33��ff�Y�C\޸                                    Bxc?�.  "          @��\��\)��z�Q�����C]�f��\)��=q�:�H���HCX�R                                    Bxc?��  �          @�������{��(��33C`h������p��Z=q��G�CZz�                                    Bxc?�z  �          @���������R��\�p��C`�)������\)�1G���z�C\��                                    Bxc?�   T          @��X����z�?��HAe�Cs�X����=q������\Ct\)                                    Bxc?��  �          @���a���
=?�z�A@(�Cs(��a���׿�
=�$��CsW
                                    Bxc?�l  �          @��\�c33�أ�?�\)A?
=Cr\)�c33��=q�����"�HCr�=                                    Bxc?�  
�          @�ff�{���(��L�;���CpG��{���\)�;����Cm�                                     Bxc@�  �          @�ff�G���  ?5@�p�Cv�R�G���(�������\Cu��                                    Bxc@^  �          A ���#33��R?�Q�A%p�C{��#33���
�����Q�Cz�f                                    Bxc@"  "          Aff�<����p�?�p�AC�CxT{�<����R���
�-�Cxp�                                    Bxc@0�  
�          A (��@  ��\)?�z�A?33Cws3�@  ��Q�\�.�RCw��                                    Bxc@?P  �          @�ff�qG��ҏ\?�G�AffCp0��qG���  �����@��Co�f                                    Bxc@M�  
�          @����u�У�?�\)A33Co}q�u��z�ٙ��L��Co�                                    Bxc@\�  
�          @����`  ��Q�?��
A5G�Cr���`  ���ÿ�
=�)p�Cr�3                                    Bxc@kB  
�          @����z����=���?@  Ck����z����
�$z����RCi8R                                    Bxc@y�  �          @��H��33��33�   �tz�Cm!H��33��=q�Fff����Ci�)                                    Bxc@��  T          @�\��  ��{�L�;�33Cn0���  ���H�-p���Ck�H                                    Bxc@�4  T          @���tz���  ��\�w
=Co���tz���ff�I����\)Cl&f                                    Bxc@��  �          @�ff�dz���\)�8Q���p�Cr��dz����\�\(����
Cnz�                                    Bxc@��  �          @���r�\�Ϯ�E���(�Co���r�\���H�X����
=Ckٚ                                    Bxc@�&  T          A   �{��ۅ����  Cp5��{���
=��  ���Ck��                                    Bxc@��  �          @�p����
�θR����^=qCmu����
��33��z��{Cg{                                    Bxc@�r  �          @�Q����R��=q�
=��  Ck8R���R��������ffCc=q                                    Bxc@�  �          A z������
=�P  ��\)Cj����������p��*�C`�                                    Bxc@��  
�          A ����Q���(��S33��(�Cl����Q���\)�����-�HCb�{                                    BxcAd  
�          @��
�g
=��\)�c�
���Cn���g
=��  ��ff�:p�Cc��                                    BxcA
  �          @��
��33�����3�
��Ci�)��33��\)��\)�p�C`=q                                    BxcA)�  "          @�����33�����AG����
Cf����33�z�H�����RC\n                                    BxcA8V  
�          @�{����
=�W��ȏ\Cen���g������'�\CY��                                    BxcAF�  �          @��R�����ff�[��̏\Cep�����dz���33�)Q�CYz�                                    BxcAU�  "          @��R��{�����e��Q�CgT{��{�g
=��G��0G�C[
                                    BxcAdH  
�          @�
=��Q���{�j�H�ۮCfQ���Q��^�R��=q�1\)CY�)                                    BxcAr�  T          @�����z������w�����Cf\)��z��QG���{�8\)CX�3                                    BxcA��  �          @�p����
���\�k���
=Cb5����
�I�����,ffCT��                                    BxcA�:  �          @�
=��G���G��aG���(�C^&f��G��<������!�CQ+�                                    BxcA��  T          A ����\)���\�p������C^����\)�9�������(=qCP�3                                    BxcA��  �          A������
=�y����{C]�3�����0  ��
=�*  COff                                    BxcA�,  �          A{������(���=q��33C]O\�����%���H�.p�CN5�                                    BxcA��  T          A�R��p������������C]@ ��p��Q�����7\)CL��                                    BxcA�x  T          A=q��\)��Q���Q��ffCY}q��\)��G���
=�={CF�{                                    BxcA�  
�          A���ff��Q���{� (�C\xR��ff�˅�˅�PG�CF�                                    BxcA��  �          A   ������
=��ff�#=qC`}q�������
�θR�X�CJ�                                    BxcBj  �          A z��L�����R��z��2�Ck�\�L����
����v��CT�                                     BxcB  �          A�\(���p������7�\Ch#��\(��޸R����u�CN�f                                    BxcB"�  �          A���l�����H�����!\)CkJ=�l���,����{�f{CX�                                    BxcB1\  "          @�p��E���
=����z�CoxR�E��1G���\)�iz�C]��                                    BxcB@  T          @�{�*=q��p�����${Cr�*=q�.�R��ff�qffCa��                                    BxcBN�  T          @��H������33�(�Cun��*=q�ָR�x�Cd��                                    BxcB]N  �          @����!G����������6��CrL��!G��p���  �)C]0�                                    BxcBk�  
�          A   �����=q��G��P��Crk���׿�p���ffCU(�                                    BxcBz�  �          A��G��R�\���w�RCn��G��#�
��ffu�C8�=                                    BxcB�@  "          @�=q�33�w
=�˅�[
=CoJ=�33������
B�CLO\                                    BxcB��  �          A �Ϳ��������
W
Cl�R����?�����HCh�                                    BxcB��  
�          A   ��=q������(��
Ce
=��=q?�=q��\){C
                                    BxcB�2  �          @�����ff�u��Rk�CO�R��ff@33����CB�                                    BxcB��  �          @�=q��S33��z��f��Cj�)�����p���C@��                                    BxcB�~  �          @���G��H���ҏ\�n33Cj&f�G����R����#�C;��                                    BxcB�$  "          @�p���7���\�}�Ci�R�>���z�aHC0p�                                    BxcB��  T          @��������H��\)�(�Cx����L���أ��o�RCj��                                    BxcB�p  
�          @���33�N�R���R�eG�Cm�)�33�&ff��  �CE��                                    BxcC  
�          @���(�?�\)�ҏ\\C�H�(�@fff��(��T��B�                                    BxcC�  T          @�\)�L��@������
�\)B�
=�L��@�G�����w�B�L�                                    BxcC*b  �          @��?+�@�\)���]�B���?+�@�33?���A�B���                                    BxcC9  �          @ٙ�?�  @�(�?L��@�z�B�aH?�  @�=q@Mp�A�ffB�.                                    BxcCG�  T          @���@���{@ϮB���C��=@���  @��BF�\C�S3                                    BxcCVT  �          @�\)@Q���@�G�B�8RC�R@Q��j=q@�  BA33C�q                                    BxcCd�  T          @��@����@�{B���C�J=@��(Q�@�=qB]
=C�Ǯ                                    BxcCs�  T          @�(�?����@�Q�B�G�C���?���n�R@�\)BSz�C��                                    BxcC�F  "          @θR@�(�@  @Z�HB�A��@�(�?B�\@���B(z�AG�                                    BxcC��  �          @���@AG�@P  ��
=�A  B<z�@AG�@���U���HBi�                                    BxcC��  "          @�=q@G
=@S�
��33�/=qB;{@G
=@�
=�0  ��Bb��                                    BxcC�8  
Z          @�  @Z�H@b�\�\(����B7�
@Z�H@��H��=q����BU{                                    BxcC��  T          @��@tz�@A��.�R��B�@tz�@u��{�b{B4�                                    BxcC˄  �          @���@��@
=q?�p�APQ�A��@��?��
?��HA�ffA���                                    BxcC�*  T          @��H@��þ�@j�HB�C�'�@��ÿ\@UB��C���                                    BxcC��  T          @��@~{�fff@~{B1{C��q@~{�@U�B\)C��q                                    BxcC�v  U          @��@�p��L��@`  BG�C��@�p��\@J=qB=qC���                                    BxcD  �          @ȣ�@����@S33A��C�Y�@��p�@*=qA�ffC�u�                                    BxcD�  �          @�z�@�
=�.{@1�A�Q�C��f@�
=��
=@�
AΏ\C��\                                    BxcD#h  "          @���@=p���G�@XQ�BC{C��{@=p�����@EB-�C��=                                    BxcD2  
�          @�  @g���=q@^�RB*C���@g���
@4z�B33C���                                    BxcD@�  �          @���@�=q��R@7
=B=qC��H@�=q��z�@�HA�\C���                                    BxcDOZ  "          @��@����#�
@8Q�BffC�� @�����
=@�A�  C��H                                    BxcD^   T          @�  @)����  @N�RB?(�C�aH@)����@!�B��C��                                    BxcDl�  T          @�
=@��ÿE�?�A�ffC�� @��ÿ��?�G�A{�C�H�                                    BxcD{L  �          @���@K����H@�A���C�ٚ@K����?���A���C�w
                                    BxcD��  �          @��\@xQ쿼(�?�ffA���C��=@xQ���?�\)Af=qC�1�                                    BxcD��  �          @���@L(����?�
=A�{C���@L(���R?��Aqp�C�
                                    BxcD�>  �          @��@hQ��
�H?�
=A���C���@hQ��(��?@  A�
C�H                                    BxcD��  �          @�Q�@���-p�?���A�(�C��@���L��?5@�C��                                    BxcDĊ  "          @���@��R���H@'�A��
C��@��R�#�
?�ffA��\C�T{                                    BxcD�0  �          @��
@qG���  @��A��C�Ф@qG��33?�\A�p�C��R                                    BxcD��  "          @�  @B�\�   @��A�  C�J=@B�\�Mp�?�ffA�
=C��                                     BxcD�|  	�          @��@%��,(�@y��B9{C��@%��{�@(Q�A��
C���                                    BxcD�"  T          @�
=@+��:�H@��B=�C�9�@+�����@;�A�ffC��\                                    BxcE�  T          @��@�\�fff@�z�B=�C�8R@�\��G�@?\)A��C�/\                                    BxcEn  "          @�  @   �C33@��BK��C���@   ���
@Y��BC�9�                                    BxcE+  
�          @�(�@<���hQ�@�G�B*(�C��{@<����ff@*�HA���C�n                                    BxcE9�  "          @��@�p�@�\)Bmp�C���@��(�@�{B'(�C��                                    BxcEH`  "          @�@J�H�=q@�G�BNC�j=@J�H���\@tz�B�HC��3                                    BxcEW  �          @�ff@~�R�8Q쿪=q�r�RC�
=@~�R�����\��C��f                                    BxcEe�  T          @��H@����$z��2�\���
C�޸@��Ϳ����a���
C�K�                                    BxcEtR  T          @�  @��\�{�J=q�
=C�^�@��\��z��u�(�C��                                    BxcE��  �          @�33@|�Ϳ����[����C���@|�Ϳz��y���1ffC��3                                    BxcE��  
�          @���@r�\��{�n{�%�C���@r�\������z��=�\C�                                    BxcE�D  
�          @��@Tz���
���CG�C�� @Tz�=��
��ff�U
=?���                                    BxcE��  �          @���@=p���(��n{�Ep�C��\@=p�>L���z=q�Sp�@z�H                                    BxcE��  	�          @�p�@)���#�
�q��[��C�Q�@)��?��H�e��K�A�=q                                    BxcE�6  �          @�
=>�z�O\)>B�\ABffC���>�z�Q녽����
C��                                    BxcE��  �          @��Ϳ�p����
@�(�B�ǮCO�{��p��-p�@�{BM�Ci�{                                    BxcE�  T          @�G��G����\@�B}
=CL.�G��-p�@��BH�Cf
                                    BxcE�(  "          @��
�	����{@�{B��CTn�	���Mp�@�33BG��Cl@                                     BxcF�  T          @1�?��H?�Q�B�\����B  ?��H?��׾�z�����B�\                                    BxcFt  T          ?���?ٙ�?B�\>��RA�
A�Q�?ٙ�?��?\)A�{A�33                                    BxcF$  T          ?�\)?p��>�z�>�@���A���?p��>aG�>k�AR�RAVff                                    BxcF2�  
�          ?���?���?5=���@i��A�\)?���?!G�>�{AK�
Aî                                    BxcFAf  
�          ?��H?�{?s33��{�$��A��R?�{?�G�<��
?�\B z�                                    BxcFP  �          @�H@��?\)�(��m�A]@��?=p��\�p�A���                                    BxcF^�  �          @Vff@C�
��Q쿕����C��H@C�
=�\)��p����R?���                                    BxcFmX  �          @N{@0  >L�Ϳ������@�\)@0  ?#�
������AR�R                                    BxcF{�  
�          @g
=@��@0�׿#�
�$  BM��@��@3�
>�p�@��
BO��                                    BxcF��  T          @�  @'
=@J=q?E�A*�\BI��@'
=@,(�?���A�B7�R                                    BxcF�J  �          @�=q?�G�@��׿(�����B���?�G�@�  ?G�A"{B�k�                                    BxcF��  �          @�ff?ٙ�@l��?��A��B���?ٙ�@9��@4z�B��Bn�                                    BxcF��  "          @�\)�B�\@��@k�Ba33B��3�B�\?��\@��B��fBȸR                                    BxcF�<  "          @�����?�  @�
=B��3C�f��녾�@��B�G�CA�=                                    BxcF��  T          @�z��\)?Q�@��B�{C33��\)�^�R@�\)B��CP33                                    BxcF�  T          @:=q?B�\?�=q<��
?z�B��=?B�\?�p�?\)A�\)Bz�                                    BxcF�.  
�          @�ff@z�?�\)�9���?p�Bff@z�@!G��
�H��BJ��                                    BxcF��  	�          @�@!�?�������l{A�33@!�@A�����7G�BH�R                                    BxcGz  T          @���@"�\?�  ���
�_�A�\@"�\@9���e�+BCff                                    BxcG   �          @�p�@,��?�
=�`���C�A���@,��@"�\�3�
�Q�B,�                                    BxcG+�  
(          @�(�@:=q?�=q�G��-{A�\@:=q@#33������B%{                                    BxcG:l  
*          @�=q@b�\?��H��R��A���@b�\@p���=q��B                                       BxcGI  �          @��H@]p�?�G�����33A�  @]p�?�G���33���A�                                    BxcGW�  
�          @��?޸R?���vff�o��B�
?޸R@'
=�I���4ffB`�H                                    BxcGf^  �          @�
=@5?�
=�L(��8
=A�{@5@(��&ff���BQ�                                    BxcGu  �          @�ff@?\)?У����
�I�
A�\@?\)@:�H�U��(�B1p�                                    BxcG��  
(          @��H@5?�33��(��P�Bz�@5@QG��^�R�G�BD(�                                    BxcG�P  �          @�{@!G�@���p��IQ�B,��@!G�@hQ��HQ��=qB\�                                    BxcG��  "          @��@"�\@���c33�:{B#�@"�\@QG��&ff� �BPp�                                    BxcG��  
�          @�p�@��?�\�l���R�B�\@��@:=q�8�����BS�                                    BxcG�B  T          @��@
=?��u��W
=B$z�@
=@AG��?\)�Q�B[��                                    BxcG��  �          @���?�Q�?�����z��BN�\?�Q�@Mp��hQ��7�B�                                    BxcGێ  
�          @�?\?�\��33�|�BE�?\@L���p  �:��B�=q                                    BxcG�4  
�          @�
=?У�@333�o\)�DQ�BoQ�?У�@xQ��&ff��G�B��                                     BxcG��  �          @��
@
=q@Dz�����
B[ff@
=q@l�Ϳ��
���Bn�                                    BxcH�  T          @��@\)?L������i�A�{@\)@	���o\)�B�B#�                                    BxcH&  
�          @�ff@h��?��,���
=A���@h��@-p���z����B�\                                    BxcH$�  	�          @�G�@4z�?�
=�>�R�$=qB	��@4z�@3�
����B3z�                                    BxcH3r  T          @�\)@��?�  �Tz��A  B�@��@0  �#�
��\BC                                    BxcHB  
�          @�z�@�R?�Q��w
=�P��B	  @�R@7
=�Fff�G�BD\)                                    BxcHP�  
�          @�ff?��
?�����(�  B=q?��
@0���|���I�Bd��                                    BxcH_d  
Z          @�ff?�p�?�{��p�� B(\)?�p�@.�R�����a�B���                                    BxcHn
  �          @��
?�
=?�=q�����~z�Bp�?�
=@HQ�����D=qBi�                                    BxcH|�  
�          @�  ?�
=?xQ���z�ǮB��?�
=@%�����c�
Bt(�                                    BxcH�V  
�          @�=q?�>������¥k�B&�\?�@�R��=qW
B�ff                                    BxcH��  �          @�@p�?�G��e��_G�A�(�@p�@
=�C�
�4�B.�                                    BxcH��  
�          @�  ?�=q?^�R@�  B��A��H?�=q���@���B�C��3                                    BxcH�H  T          @�Q�@   ?fff@���B~
=A���@   �\)@�ffB���C�                                      BxcH��  �          @��@6ff?�z�@q�BG�RA�@6ff=�@���BZ@
=                                    BxcHԔ  T          @��
@J�H?��@r�\B?{A�\)@J�H=u@���BO  ?�                                      BxcH�:  �          @�Q�?.{?W
=@�z�B�G�BL�R?.{�Y��@�z�B�33C�W
                                    BxcH��  "          @�33?���?333@�p�B�W
A��?����h��@�z�B�z�C��f                                    BxcI �  
�          @���@�R?.{@�
=B~A�  @�R�Y��@�{B|
=C��f                                    BxcI,  �          @��
@6ff?�{@��\BW��Ạ�@6ff���
@�G�Bg��C�,�                                    BxcI�  T          @���@C�
?�33@~�RBCz�A�33@C�
>��R@���BY@��                                    BxcI,x  "          @�\)@dz�@*�H?aG�A3�B{@dz�@  ?�
=A�B ��                                    BxcI;  �          @�
=@j=q@N�R?�p�A�G�B%��@j=q@%@�HA��B\)                                    BxcII�  
�          @�33@ff@\���3�
��\B_
=@ff@����=q��(�Br(�                                    BxcIXj  �          @�=q?�(�@~�R�z=q�*��B�u�?�(�@�  �p���=qB�B�                                    BxcIg  �          @�{?�Q�@����G
=��\B�\)?�Q�@�녿�{��=qB���                                    BxcIu�  "          @�@*=q@|(��'
=��(�B_�
@*=q@�녿�  �U�Bn�H                                    BxcI�\  
�          @��H@
=q@����O\)�\)B|@
=q@�녿�  ����B��)                                    BxcI�  
�          @�
=@.{@����z����\Buff@.{@�Q������Bzp�                                    BxcI��  "          @�(�@0��@�=q��\��G�Bq  @0��@�(��B�\��=qBw
=                                    BxcI�N  T          @�=q@
=q@�(��.{��
=B�Q�@
=q@��H?p��A=qB�                                      BxcI��  "          @��?�@�\)�����A��B���?�@�33>�@���B�ff                                    BxcI͚  
(          @��@�@�  ���Ϳ���B���@�@�G�?�Q�As\)B��                                    BxcI�@  "          @���@E@���?��A�p�B^�H@E@s33@<��A���BK��                                    BxcI��  "          @���@-p�@�z�?333@�33Bt{@-p�@�ff@
=qA�z�Bjz�                                    BxcI��  
�          @���@:�H@��?n{A�Bm33@:�H@�z�@Q�A£�Ba�                                    BxcJ2  �          @�(�@)��@��@O\)B�\Bj�@)��@J=q@��B;��BH
=                                    BxcJ�  �          @�  @��@~�R@o\)B�RBi�R@��@)��@���BS��B=Q�                                    BxcJ%~  
�          @�  ?�Q�@9��@�  BP�
B`��?�Q�?�33@�p�B�\)B(�                                    BxcJ4$  "          @�ff@�@>�R@���BF\)BZ
=@�?��@�  By=qB��                                    BxcJB�  �          @��@#33@8��@�Q�B8z�BB33@#33?�ff@�ffBfG�A�\)                                    BxcJQp  �          @���@^{@:=q@aG�B�B �@^{?�(�@�\)B>ffAң�                                    BxcJ`  �          @�{@dz�@9��@n�RB
=Bz�@dz�?�33@�BA�\A�Q�                                    BxcJn�  "          @���@y��@8��@K�BB(�@y��?�@y��B(AƸR                                    BxcJ}b  �          @���@i��@#�
@s�
B"(�B{@i��?�ff@��BB��A��H                                    BxcJ�  
�          @�33@dz�@�@a�B �A��
@dz�?��
@���B=�A���                                    BxcJ��  T          @���@u@,(�@C�
B�B{@u?�z�@n{B&�A���                                    BxcJ�T  �          @�G�@@  @@~{B8��B�@@  ?��@�\)BZp�A��                                    BxcJ��  �          @�ff@��?O\)@X��B��A#33@���k�@^�RB
=C��H                                    BxcJƠ  �          @�z�@��>���@c�
B�@��\@�녿+�@aG�B�\C���                                    BxcJ�F  �          @��\@�@3�
@��A��
B�R@�@�
@7�A�
=Aљ�                                    BxcJ��  "          @��@�p�@g
=?�Q�Ai�B#\)@�p�@A�@(�A�ffB                                    BxcJ�  T          @�@~{@x��?�33A;�B1�@~{@XQ�@\)A�=qB!�                                    BxcK8  T          @��@�G�@^{?}p�A"=qB�R@�G�@A�?��HA���B                                    BxcK�  T          @��@�ff@<��?}p�A(��B(�@�ff@!�?�A��A�
=                                    BxcK�  
�          @�ff@�@A�?+�@��B	�\@�@-p�?\A�z�A�p�                                    BxcK-*  "          @�p�@��
@[�>���@P  B�@��
@L(�?��A]�B�                                    BxcK;�  �          @��@n�R@H��?��AI�B �@n�R@,(�?��HA�Q�BG�                                    BxcKJv  T          @�\)@k�@"�\@&ffA��B
�@k�?�
=@N{B
=Aď\                                    BxcKY  "          @���@p  @(��@$z�A�Bp�@p  ?��@N{B��Aˮ                                    BxcKg�  �          @��@�ff@A�?��A=p�B(�@�ff@%?�z�A�\)A��                                    BxcKvh  
�          @�z�@��@U?
=q@���B�H@��@B�\?�p�A\)B��                                    BxcK�  
�          @�  @^{@S33�#�
��B.G�@^{@J�H?h��A.�RB)                                    BxcK��  "          @��@�=q@�
?�
=A�33A�\@�=q?�z�@ ��A홚A��
                                    BxcK�Z  T          @��\@��\@.{?�
=AY��B�R@��\@G�?�z�A�{A��                                    BxcK�   T          @���@w�@	��@�A�A�Q�@w�?�z�@333B�\A��\                                    BxcK��  T          @�=q@j�H?aG�@7�B  AW�
@j�H<�@@  B�?                                       BxcK�L  �          @���@B�\���R@fffB>��C��@B�\���@G
=BffC�
=                                    BxcK��  T          @�\)@dz῝p�@|��B9  C�w
@dz��33@\(�B  C��                                     BxcK�  "          @�ff@�z�fff?�z�A��
C��@�zῳ33?�=qA��HC��\                                    BxcK�>  "          @��
@s�
=�@3�
B�?�  @s�
�.{@.�RBC���                                    BxcL�  �          @��@%��\)@s33B^33C���@%��=q@dz�BK33C�n                                    BxcL�  "          @�(�@AG�<��
@s33BM��>�p�@AG����@i��BC=qC��                                     BxcL&0  T          @��@B�\?#�
@mp�BH(�A=G�@B�\��
=@o\)BJp�C��                                    BxcL4�  �          @�z�@'
=>W
=@hQ�BX@�p�@'
=�L��@c33BQ��C�u�                                    BxcLC|  T          @�@q녾.{@(��B�RC�� @q녿c�
@\)B�
C�`                                     BxcLR"  �          @��R@�
=�8Q�@��A�C�'�@�
=����@ffA�
=C�)                                    BxcL`�  �          @�p�@xQ쿪=q@/\)B��C��@xQ���\@��Aڏ\C�*=                                    BxcLon  
�          @��@c�
���@@  Bz�C�w
@c�
���@%B�C��                                    BxcL~  �          @��@�=q���
@�RAمC���@�=q��?�p�A��
C�y�                                    BxcL��  �          @�@u��z�H@7�B�C�ٚ@u���(�@   A���C���                                    BxcL�`  "          @���@�p���ff>��@�C�Y�@�p���\)<�>���C���                                    BxcL�  �          @��R@�\)��p�?333Az�C�Z�@�\)��\)>�{@��
C���                                    BxcL��  "          @���@�
=���?
=@�=qC�N@�
=��
=>�\)@]p�C��{                                    BxcL�R  �          @�p�@�  ��z�?�\@�  C��@�  ��G�>8Q�@
=C��3                                    BxcL��  T          @�ff@�ff��z�?��RAb{C���@�ff��Q�?O\)A33C�
                                    BxcL�  T          @��H@�\)��G�?\A�  C�AH@�\)�
=?}p�A4z�C�g�                                    BxcL�D  
�          @��@���\)@z�AŮC��@���H?�p�A��C�Q�                                    BxcM�  T          @���@����  ?�\)A�=qC�T{@����?��A6ffC�}q                                    BxcM�  �          @�{@���Q�?ٙ�A��\C��@���?�G�AV�RC�Ǯ                                    BxcM6  T          @���@�Q�Ǯ?�Q�A���C�Y�@�Q����?��HAK33C�Y�                                    BxcM-�  T          @�=q@�33�n{?��A��\C���@�33���?ǮA�C��q                                    BxcM<�  �          @�
=@�(���
=?��A�Q�C��@�(��aG�?�{A�(�C��=                                    BxcMK(  "          @�@��׿&ff?z�HA7�C��@��׿c�
?E�AQ�C���                                    BxcMY�  �          @�p�@�p���p�?�  A1C��R@�p�����?#�
@�\C���                                    BxcMht  
�          @��R@�  ��z�?���Aj�\C�g�@�  ��Q�?W
=A�
C��                                    BxcMw  T          @��H@�Q쿮{?�AL  C�j=@�Q��\)?G�AffC��                                    BxcM��  "          @�33@�zῷ
=?��RA��RC��q@�z��\?���A9C��                                    BxcM�f  
�          @��@�p���  ?��\AU�C�>�@�p���p�?8Q�A
=C�                                    BxcM�  T          @�G�@�����z�?W
=A+�
C�q�@������?�\@�Q�C�W
                                    BxcM��  �          @���@��׿�p�?aG�A"{C���@��׿�?�@��HC���                                    BxcM�X  T          @�z�@�Q쿊=q>��@���C���@�Q쿓33=�G�?�G�C��=                                    BxcM��  
�          @���@�=q���H@%A�\)C�  @�=q��@Q�AǅC�xR                                    BxcMݤ  T          @�@��Ϳ�\@
=A��C���@����  ?�=qA�{C�R                                    BxcM�J  "          @��@�{��R?���Ad  C��)@�{� ��?B�\@���C���                                    BxcM��  	�          @�z�@�{��p�?!G�@���C�R@�{�z�=�G�?��C���                                    BxcN	�  	�          @��\@��ÿ�Q�?�R@�33C�!H@��ÿ��>.{?ٙ�C���                                    BxcN<  "          @��@��R���>�G�@�
=C�  @��R��33=�Q�?\(�C�˅                                    BxcN&�  
�          @�(�@�Q쿌��>�\)@6ffC�XR@�Q쿑논#�
��Q�C�+�                                    BxcN5�  �          @��@�Q쿜(�>�\)@7�C��@�Q쿠  ����33C��)                                    BxcND.  
�          @�(�@�p����
>\)?�33C�q�@�p��\��  �=qC���                                    BxcNR�  �          @�@������G����C�\@���(��   ����C�q�                                    BxcNaz  �          @�\)@��H���\>L��?��RC���@��H���
����  C���                                    BxcNp   "          @��R@�z�\(����Ϳ}p�C���@�z�O\)���R�HQ�C���                                    BxcN~�  "          @�p�@��H�^�R�����C��R@��H�O\)��{�_\)C��                                    BxcN�l  
�          @�p�@���(�>L��@�
C��@���#�
=#�
>�ffC��)                                    BxcN�  �          @��@�z��녾aG���RC��@�zᾳ33���
�Mp�C�=q                                    BxcN��  �          @��\@���=#�
���H��G�>�ff@���>#�
������?�
=                                    BxcN�^  T          @�33@�\)?u�W
=��A.ff@�\)?z�H<�>�p�A2ff                                    BxcN�  	�          @���@���?�  �k��$z�A4��@���?��
<��
>�=qA9G�                                    BxcN֪  �          @���@�  ?
=���R�W�@θR@�  ?&ff�#�
��
=@�\                                    BxcN�P  "          @�@�p�?�=q=��
?aG�A�{@�p�?�G�>�@�z�A�Q�                                    BxcN��  �          @�Q�@��?�\)���
�k�A�(�@��?���>��R@\(�A�                                    BxcO�  
�          @�Q�@�=q?���=#�
>ǮAs33@�=q?�=q>�p�@�G�Aj�R                                    BxcOB  �          @��@�33�Ǯ?��@��
C��@�33��>��H@�p�C�8R                                    BxcO�  �          @�ff@�z���H?�@�  C�e@�z���>Ǯ@��\C��                                    BxcO.�  
�          @��@����.{?W
=A�C��@�����Q�?E�Az�C�
=                                    BxcO=4  
(          @�(�@��>��?z�@ȣ�?��@��<#�
?��@�\)=�\)                                    BxcOK�  
Z          @���@�
=��?B�\A33C�z�@�
=�!G�?�R@�Q�C���                                    BxcOZ�  
Z          @�G�@�(��=p�?uA((�C��H@�(��s33?B�\A�
C���                                    BxcOi&  T          @��R@����
=?��
A7�C���@����Q�?\(�A��C�ff                                    BxcOw�  �          @�
=@���k�?��
A5��C���@�����?p��A&=qC�^�                                    BxcO�r  
�          @�{@��ͿTz�?��RA�33C�/\@��Ϳ�z�?�  A^ffC�U�                                    BxcO�  �          @�{@��R�fff?�APQ�C��q@��R��33?k�A#�
C�y�                                    BxcO��  "          @�p�@�\)�&ff?�  A_
=C�G�@�\)�n{?��A<��C���                                    BxcO�d  "          @�
=@�(��z�H?ǮA�G�C�P�@�(�����?��
Ad(�C�k�                                    BxcO�
  �          @��H@�Q�aG�?˅A���C�@�Q쿝p�?��Ahz�C�)                                    BxcOϰ  %          @���@����(�?�  A]G�C���@����c�
?���A=G�C��                                    BxcO�V  
�          @�
=@���W
=?�  A]�C�1�@����{?��\A4  C���                                    BxcO��  T          @��R@���8Q�?�G�A�\)C���@�����?�ffAh  C��                                    BxcO��  T          @�
=@�=q�!G�?��A���C�J=@�=q��ff?�
=A�z�C�޸                                    BxcP
H  �          @�=q@��\��@Q�A��C��@��\��  ?���A��RC�&f                                    BxcP�  �          @���@��\�.{?��RA�C��R@��\����?��
A�  C�o\                                    BxcP'�  �          @�(�@�  ��z�@�RA��
C�"�@�  ���?�33A��C�t{                                    BxcP6:  T          @��@�(���z�@��A�=qC���@�(����?��A�ffC��                                    BxcPD�  �          @�=q@�(��˅@�RA�  C�f@�(���@33A�z�C��                                    BxcPS�  �          @�(�@�ff����?��
A���C���@�ff��G�?�(�A~�\C��f                                    BxcPb,  "          @��H@��R��?xQ�A(  C�
@��R�=p�?Q�AffC��
                                    BxcPp�  
�          @�ff@�z�>�
=>���@`��@�ff@�z�>���>�
=@�@a�                                    BxcPx  �          @��@��\?���:�H��\Abff@��\?�(�������RAw\)                                    BxcP�  T          @��H@��@"�\��Q����\A��@��@:=q����\(�B�
                                    BxcP��  �          @��@��
@����6�HAͅ@��
@!녿
=q���\Aۅ                                    BxcP�j  �          @���@�@��(���p�A�Q�@�@7��˅���
B�\                                    BxcP�  "          @��@�z�@�Ϳ�z��lz�A�  @�z�@p��aG���RA�                                      BxcPȶ  �          @���@���@\)��=q�4��Aۙ�@���@*�H�   ���A���                                    BxcP�\  T          @�(�@���@1G��^�R��A�{@���@8�þk��(�B�\                                    BxcP�  
�          @�33@�z�@7���  �,��B�H@�z�@AG������eB
Q�                                    BxcP��  "          @�@��H@Dz������33B(�@��H@E>�z�@FffB��                                    BxcQN  "          @�ff@�ff@-p��\)����A�G�@�ff@*�H>��@�p�A�\                                    BxcQ�  "          @�@��@S�
�}p��)��B\)@��@\(��k��{B�H                                    BxcQ �  
�          @�
=@w
=@N{��Q���z�Bff@w
=@aG��p���"�\B)G�                                    BxcQ/@  W          @��@2�\@��R���
����Ba��@2�\@�  �O\)�
�\Bh�                                    BxcQ=�  Q          @���@�{?�=q�fff��\A��@�{?޸R������A��
                                    BxcQL�  
�          @�Q�@��
?��\(��z�A��@��
?�p���G���33A�G�                                    BxcQ[2  "          @��@�G�@%��!G���(�A�z�@�G�@)���L�Ϳ�A�                                      BxcQi�  	`          @��@��?���L���	��A��@��?��>\)?��RA�ff                                    BxcQx~  "          @�z�@�G�?&ff�+���(�@���@�G�?G���\��{Aff                                    BxcQ�$  
�          @�Q�@�=q?fff�z�H�,z�A Q�@�=q?���B�\��AAp�                                    BxcQ��  �          @�  @��?�{��z��K�AC�@��?�녽#�
���HAI                                    BxcQ�p  T          @��H@�?xQ�L���
=qA)G�@�?�\)����AB�R                                    BxcQ�  �          @��@�Q�?�G��#�
����A���@�Q�?�(�>�p�@~�RA�p�                                    BxcQ��  
�          @��\@�ff?��þ\)��G�A9@�ff?���=�Q�?�  A:�R                                    BxcQ�b  "          @��@���>��(����@���@���?z�   ���H@��                                    BxcQ�  "          @�G�@��R?8Q�W
=��\@���@��R?@  �L�Ϳ\)A33                                    BxcQ��  �          @���@�ff>��H?0��@�@�(�@�ff>���?G�A\)@j=q                                    BxcQ�T  �          @��\@�  ?�?(�@�=q@�p�@�  >�(�?5@�
=@�z�                                    BxcR
�  "          @�=q@�{?��
?   @�ffA333@�{?fff?5@�A�                                    BxcR�  �          @�Q�@��?\(�>�
=@�=qA33@��?@  ?
=@�ffA��                                    BxcR(F  �          @�=q@���=L�;�=q�;�?�@���=��;�  �0��?�\)                                    BxcR6�  �          @��@��׾���������C�c�@��׾8Q����ffC�                                    BxcRE�  T          @��@���>aG��Ǯ��Q�@Q�@���>�����{�j�H@N{                                    BxcRT8  T          @���@�  >B�\�z����H@
=@�  >��R����Q�@X��                                    BxcRb�  �          @���@��?�{���ǮAC�@��?��������e�AS�
                                    BxcRq�  �          @���@�{?�p��n{�#�A�
=@�{?�녿�����
A���                                    BxcR�*  �          @�(�@��?��׿��H�g�A�=q@��?��Ϳfff�+\)A��\                                    BxcR��  �          @�p�@�33��=q��{����C��@�33�+���\�ǅC��)                                    BxcR�v  �          @��@��>Ǯ���R�33@�
=@��?#�
�����hz�A
=                                    BxcR�  T          @�{@��?\)�.{��H@�G�@��?.{�����z�A��                                    BxcR��  T          @��R@���>��W
=�8��@�\)@���>��H��G���G�@�                                    BxcR�h  "          @c�
@(�����(��aG�C�H�@(����
�!G��g�C��=                                    BxcR�  "          @xQ�@333@%>��@�\)B+33@333@(�?z�HAm��B$p�                                    BxcR�  �          @�33@Z�H@{?8Q�A�HB33@Z�H@G�?��HA���B=q                                    BxcR�Z  
�          @g
=?����p�>��A���C�c�?����ff>\A��HC�XR                                    BxcS   T          @����p���33���Ϳ�Q�Cp� �p�����z�H�5�Co�)                                    BxcS�  �          @�=q�O\)�q녾�G�����Cec��O\)�g���Q��\Q�Cd)                                    BxcS!L  �          @�ff�K��X�þ�\)�_\)Cb�=�K��P�׿u�>{Ca�3                                    BxcS/�  
�          @����AG��P��?+�A	��Cc:��AG��U�    =L��Cc��                                    BxcS>�  �          @��
�0���g
=?
=q@�=qCh���0���i���B�\�Q�Ch��                                    BxcSM>  �          @���p��I���s33�V=qCh��p��9���Ǯ���RCe�R                                    BxcS[�  "          @�{� ���k�=���?�=qCk�q� ���hQ�!G��ffCkY�                                    BxcSj�  
�          @��R�,(��S33�u�J�HCf�=�,(��B�\�˅��\)Cd��                                    BxcSy0  �          @����/\)�E�0����HCdn�/\)�8�ÿ��
���HCb��                                    BxcS��  "          @��Ϳ��5�@��B!�
C��\���P  ?���A��C�`                                     BxcS�|  �          @���@U�����?�z�A�Q�C�7
@U���?�{A�
=C���                                    BxcS�"  
�          @��\@c33����?�z�A�\)C�aH@c33�0��?�A�ffC�w
                                    BxcS��  T          @��
@u���
@ ��A�Q�C��H@u�5?�z�A�Q�C��=                                    BxcS�n  "          @�=q@�z�>���?�{AV�R@��R@�z�>8Q�?�Ab{@(�                                    BxcS�  
(          @�
=@�  ?xQ�?=p�A�RAN�R@�  ?O\)?k�A?�
A,��                                    BxcSߺ  
�          @���@�G�?J=q?�33Az�\A0��@�G�?��?��A�  @�                                      BxcS�`  
%          @�G�@r�\>u?�{A�(�@dz�@r�\��?���A��\C��                                     BxcS�  T          @r�\@[�?(�?�(�A���A (�@[�>��R?ǮA£�@��                                    BxcT�  
�          @j=q@QG�?���?�  A�\)A�\)@QG�?W
=?�Q�A��HAg33                                    BxcTR  �          @i��@P��?�G�?�ffA���A���@P��?��?��\A�G�A�\)                                    BxcT(�  
�          @�Q�@I��@33?\)A(�B(�@I��?�33?s33Af�RA�
=                                    BxcT7�  
�          @��
@`��?�  ?�G�A��HA�\)@`��?p��?�p�A�\)Ap(�                                    BxcTFD  T          @�z�@k�?&ff?�  A���A�@k�>���?�A�{@��                                    BxcTT�  �          @��@fff?O\)?ٙ�A��
AJ{@fff>��?���A�G�@���                                    BxcTc�  T          @�z�@e?�G�?��
AˮA|  @e?(��?�Q�Aߙ�A%                                    BxcTr6  T          @��@dz�>�@�A@���@dz�=u@Q�A�  ?}p�                                    BxcT��  �          @���@mp�?��?�\A��A�p�@mp�?=p�?���A�{A4                                      BxcT��  �          @��
@z=q?�p�?�G�A��A�\)@z=q?k�?��HA��RAS�
                                    BxcT�(  �          @���@��H?}p�?��HA���AX��@��H?&ff?�{A��
A\)                                    BxcT��  �          @�=q@�p�?\)?�A��\@�  @�p�>W
=?��A�(�@0                                      BxcT�t  "          @�\)@�Q쾞�R@A���C�H@�Q�333@   A�33C��3                                    BxcT�  
�          @�@�Q���@p�A�C�8R@�Q쿆ff@�
A�(�C�s3                                    BxcT��  �          @��\@��׾��@,��B �RC���@��׿xQ�@$z�A��C���                                    BxcT�f  "          @�Q�@�(���Q�@2�\B�C�|)@�(��aG�@*�HB�C��                                    BxcT�  �          @�Q�@��
�k�@A���C�z�@��
�(��@��A�(�C���                                    BxcU�  
�          @��\@�33�8Q�@�A�(�C�Ф@�33�\)@ ��A��HC�P�                                    BxcUX  �          @��@��׿.{?�  A�C���@��׿}p�?˅A�=qC��\                                    BxcU!�  
�          @��H@�G��\)@\)A�Q�C�t{@�G��xQ�@A�(�C��)                                    BxcU0�  "          @�G�@�{�k�@&ffA�p�C��\@�{����@Q�A��HC��                                     BxcU?J  �          @�G�@�33��
=@ ��A��
C�ff@�33�G�?�33A��HC�/\                                    BxcUM�  
�          @�@�ff��=q@A�z�C�<)@�ff�&ff@   A�C���                                    BxcU\�  
�          @��R@��p��@ ��A�33C��@����?�ffA��\C��H                                    BxcUk<  �          @��R@�{�aG�@�A��C�j=@�{��p�?�=qA��C�>�                                    BxcUy�  T          @���@�{���
?��HA�z�C���@�{�+�?�\)A�
=C��                                    BxcU��  
�          @�p�@�\)=#�
?�p�Aȣ�?�@�\)���
?���A��
C���                                    BxcU�.  �          @�33@�ff��?�\)A�Q�C��@�ff��p�?�A�ffC�~�                                    BxcU��  �          @���@��׾��?���A���C�  @��׾�?�\A��HC�Ф                                    BxcU�z  �          @���@�(���?���A�z�C���@�(���33?��A���C���                                    BxcU�   �          @��
@�\)��@A�{C�G�@�\)���
?���A���C�3                                    BxcU��  T          @��
@�\)���H@�
Aȏ\C��@�\)�Ǯ?�A��RC���                                    BxcU�l  �          @��H@�  �}p�@33Aə�C�j=@�  ����?�A���C�5�                                    BxcU�  �          @���@��H����@
=A�p�C��@��H��
=?�A�
=C��)                                    BxcU��  
�          @�  @��\���@
=qA��HC��3@��\��
=?�Q�A���C�W
                                    BxcV^  "          @�(�@�  �333@��A�p�C�L�@�  ���@ffA�ffC��f                                    BxcV  
�          @���@�=q���@=qA���C�  @�=q�xQ�@G�A�C�Ff                                    BxcV)�  T          @���@xQ쾊=q@#33B
=C���@xQ�8Q�@p�A���C���                                    BxcV8P  T          @�
=@|(��&ff@>{B��C�XR@|(���z�@3�
B	z�C���                                    BxcVF�  �          @��\@{���(�@=p�B��C�j=@{���(�@,��B �\C�33                                    BxcVU�  T          @��@|(���G�@9��BG�C�q@|(���  @(��A��C���                                    BxcVdB  
�          @��R@c33��33@AG�B�C���@c33�	��@+�B�C�`                                     BxcVr�  �          @�  @`  ��(�@;�B�C�O\@`  �p�@"�\A��
C�y�                                    BxcV��  T          @�z�@�Q�?W
=?�p�A��
A Q�@�Q�?�?���A�\)@���                                    BxcV�4  "          @���@���?G�@A���A��@���>���@(�AĸR@��H                                    BxcV��  
(          @��@�\)?(�@
=qA�
=@�\@�\)>aG�@�RA�{@,(�                                    BxcV��  "          @�Q�@�Q�?G�@�A��A(�@�Q�>��@��A�(�@�{                                    BxcV�&  
�          @��
@�{?�p�?��A
=A��@�{?�(�?���A��HAv{                                    BxcV��  "          @��@��\?��?�  A��RAl��@��\?Q�?�A�ffA,                                      BxcV�r  T          @�(�@��H?��\?�=qA��\A���@��H?n{@G�A�=qABff                                    BxcV�  �          @��@���?�G�@33A�Q�A�33@���?aG�@\)AٮA:�H                                    BxcV��  �          @�p�@�  ?��\@ffA�
=A��\@�  ?aG�@�\A�z�A;�                                    BxcWd  
�          @�\)@��?�G�@��A�
=A���@��?Tz�@%�A�Q�A3�
                                    BxcW
  �          @�  @���?8Q�@\)A���AG�@���>�\)@%�A�=q@r�\                                    BxcW"�  �          @�Q�@��R>��@��A��H@X��@��R��@G�A�{C�,�                                    BxcW1V  T          @��
@��?�=q?�(�A�Q�A�\)@��?��?��HA��A�p�                                    BxcW?�  �          @���@�?���?�Q�A�  Ar=q@�?c�
?�\)A�(�A6{                                    BxcWN�  
�          @�ff@���?!G�?�A�=q@�
=@���>���@   A�Q�@p��                                    BxcW]H  
�          @��H@�p�?z�H@33Aי�AHz�@�p�?z�@(�A�G�@�                                    BxcWk�  T          @��R@�z�?u@
�HA��HA;\)@�z�?z�@33AѮ@��
                                    BxcWz�  "          @���@�=q?Q�@�RA�33A"�\@�=q>�(�@A�@��                                    BxcW�:  �          @�ff@���>�?�Q�A��H@��@���>\)@   A�(�?�G�                                    BxcW��  �          @�Q�@�G�>��@z�AÅ@�33@�G�=L��@
=A�\)?�R                                    BxcW��  
Z          @��@���>���@�A�z�@tz�@��ýL��@ffAƣ�C���                                    BxcW�,  T          @�  @���>��R@
=A�p�@z=q@��׽L��@Q�AɮC��\                                    BxcW��  �          @��@��\?&ff@ffA��A(�@��\>u@�A��
@Mp�                                    BxcW�x  
�          @���@�(�?p��@,(�B�HAL��@�(�>��@3�
Bff@�Q�                                    BxcW�  
�          @��\@z=q?��@B�\BA�ff@z=q?�R@L(�B=qA�
                                    BxcW��  
�          @�  @���?c�
@3�
B	Q�AG�
@���>��@:�HBz�@�G�                                    BxcW�j  
�          @�\)@��
?z�@.�RB�@��R@��
=�G�@2�\BQ�?Ǯ                                    BxcX  �          @�{@��
>���@��A�=q@��
@��
�L��@�A���C��\                                    BxcX�  "          @�
=@��>��@  A�
=@��@��<�@�A��H>Ǯ                                    BxcX*\  
(          @���@�ff>��H@-p�Bz�@�
=@�ff=#�
@0  B�?�                                    BxcX9  T          @���@�33?�@9��B(�@�{@�33<�@<(�B�>��                                    BxcXG�  �          @���@�{>W
=@0��Bff@8��@�{��  @0��B=qC�L�                                    BxcXVN  "          @�{@���?��@.�RB��@��R@���=�Q�@1�B	��?��\                                    BxcXd�  T          @�Q�@��R>��@,(�B��@c33@��R�B�\@,(�BG�C��)                                    BxcXs�  �          @�{@��>�p�@7
=B��@��
@�녽�G�@8Q�B�C�Ff                                    BxcX�@  
�          @��\@��R<�@:=qB��>�Q�@��R��ff@8Q�B��C��                                    BxcX��  �          @���@�=q=�G�@7
=B \)?�33@�=q��Q�@5A���C��q                                    BxcX��  T          @��@��R���R@   A��C��@��R�8Q�@=qA��HC��=                                    BxcX�2  
(          @�(�@��
��  ?�33A�33C��@��
��ff?��HA��HC�'�                                    BxcX��  
Z          @���@��Ϳ�=q?��HA��\C�R@��Ϳ��?�G�Au��C��)                                    BxcX�~  �          @��@�=q���?��\AG\)C��3@�=q���?=p�A�C���                                    BxcX�$  T          @�@l���0��?8Q�A��C���@l���5>��@H��C�E                                    BxcX��  �          @��@a��&ff?fffA8��C��@a��-p�>�@���C�:�                                    BxcX�p  �          @�G�@\���.�R?fffA9�C��@\���5>�G�@�p�C�AH                                    BxcY  
(          @J=q�u�=q���H���Cx=q�u�ff���(
=Cuc�                                    BxcY�  
�          @|���Q쿓33�0  �8\)CM�3�Q�0���9���E�RCD�                                    BxcY#b  �          @���,���!��'
=���C_��,���z��?\)�$��CYh�                                    BxcY2  	�          @�p��A��*=q��
��  C]T{�A�����-p���CX��                                    BxcY@�  "          @���:=q�G��   ����Cc��:=q�0����R���C_�                                     BxcYOT  �          @�{�A��I���p���=qCb��A��0���,(��{C^B�                                    BxcY]�  �          @���<���_\)��p����Ce�
�<���HQ��!G�����Cb�3                                    BxcYl�  
�          @��\�E��S�
�����Cc�E��@  ������C`0�                                    BxcY{F  
�          @���E�8Q��Q���\)C^�R�E�!��Q���{C[L�                                    BxcY��  �          @��H�c33��׿�G����
CT�\�c33�����ff�ۮCP�{                                    BxcY��  
�          @���\(��	����
����CT��\(���ff�Q���CO��                                    BxcY�8  "          @���9���4z���H���C`E�9���{���� p�C\z�                                    BxcY��  �          @���A��6ff��z���{C_.�A��"�\����C[�q                                    BxcYĄ  �          @�p��K��$z��\)��G�CZ���K��G�� �����
CW��                                    BxcY�*  
�          @�z��@  �33�(����CY���@  ���!���RCT��                                    BxcY��  X          @�
=�Q녿�p��33���RCO�q�Q녿���#33��HCJ(�                                    BxcY�v  �          @��
�X�ÿ����"�\�
�CK&f�X�ÿ��
�.�R��CDٚ                                    BxcY�  
(          @�ff�^{=����0  �Q�C2T{�^{?
=q�,�����C+33                                    BxcZ�  "          @�(��B�\����S�
�6�CG���B�\����\(��@ffC>0�                                    BxcZh  �          @���   ���Mp��.33C^T{�   �޸R�a��E
=CV��                                    BxcZ+  �          @�z῅��\(��)�����C}0�����>�R�J=q�4Q�Cz�                                     BxcZ9�  
�          @�ff����!��P  �0z�Ce  ��Ϳ�p��g
=�J��C^�                                    BxcZHZ  
�          @��������N{�4��Cd������{�c�
�Np�C]T{                                    BxcZW   
�          @�zῃ�
�o\)�"�\�{C~�H���
�R�\�Fff�(  C|�f                                    BxcZe�  "          @�녽#�
���\��Q���\)C��H�#�
�~{�&ff�
=C���                                    BxcZtL  
�          @�=q?W
=��ff�����p�C��?W
=��G���=q���HC�=q                                    BxcZ��  �          @���>����G��W
=�p�C��f>����33��=q����C���                                    BxcZ��  �          @�33>����G��\)��ffC�y�>����zῧ��{�C��                                    BxcZ�>  "          @��
?Y�����þ�p����C�?Y�������\)�Tz�C�%                                    BxcZ��  T          @�=q?n{��{����ҏ\C���?n{�������
�w33C��                                    BxcZ��  T          @��
?�G���\)��G����C��q?�G���������hz�C��                                    BxcZ�0  
�          @�  ?u���H��R�G�C���?u�|�Ϳ�  ���HC��R                                    BxcZ��  T          @��?#�
�{��L���5p�C���?#�
�p  ��33��  C�޸                                    BxcZ�|  �          @qG�?L���c�
�k��b{C�O\?L���W���(���\)C��f                                    BxcZ�"  T          @�G�?333�r�\��Q���G�C�>�?333�c�
��\��{C��{                                    Bxc[�  	�          @w
=?\)�c�
�������
C�s3?\)�S33������C�˅                                    Bxc[n  T          @|(�?(��aG�����=qC��?(��N{�����C�\)                                    Bxc[$  �          @�\)?!G��qG������33C��q?!G��\(��
=�
=C�.                                    Bxc[2�  "          @��?&ff�j�H���R��RC�?&ff�S�
�"�\��RC���                                    Bxc[A`  T          @�{?W
=�\���{� \)C��\?W
=�C�
�/\)�#(�C���                                    Bxc[P  "          @�(�?�=q�^�R������C��f?�=q�HQ��(����C���                                    Bxc[^�  �          @�z�>�33�p  �����{C��3>�33�W
=�0  �C��                                    Bxc[mR  �          @�ff?�R�s�
�����G�C��?�R�Z�H�1G����C�*=                                    Bxc[{�  
�          @�(�>�ff�mp������RC�s3>�ff�S�
�5�� �C��q                                    Bxc[��  "          @�  >\)�b�\�ff�
=C�+�>\)�HQ��7��)�
C�Q�                                    Bxc[�D  �          @��?aG��I���{�{C���?aG��.�R�;��6ffC��                                    Bxc[��  "          @�p�@Q��{�0  �&�HC��=@Q��\�C�
�?G�C�,�                                    Bxc[��  T          @��\�8Q��W
=�$z���\C�q�8Q��:�H�C�
�9=qC�5�                                    Bxc[�6  �          @��k��`  �"�\�G�CT{�k��C�
�C33�.�C}Y�                                    Bxc[��  "          @�������fff�����Cy�{�����O\)�#�
��HCw�\                                    Bxc[�  	`          @�\)�L���H���Q���C�=�L���/\)�5�3\)C}Ǯ                                    Bxc[�(  
�          @��\��33�U�����ׅCz���33�@����\�Q�Cy{                                    Bxc[��  �          @��
��(��Vff��33���
Cvh���(��AG�����\)Ct)                                    Bxc\t  "          @�=q�˅�N{��33�ޣ�Cs��˅�8�������Cq+�                                    Bxc\  �          @�(���\�L(���z�����Cp���\�7
=�Q��(�Cn(�                                    Bxc\+�  
�          @��R���Z�H���R���Cn�����L(���  ���
Cl�)                                    Bxc\:f  �          @�\)���R�Vff�˅��  CoW
���R�C�
���\Cm
=                                    Bxc\I  �          @��ÿ�
=�U������p�Co����
=�@���ff�(�CmG�                                    Bxc\W�  
�          @����H�H�ÿ����Q�Cn����H�3�
�
=�	�HCk
                                    Bxc\fX  
�          @��\�
=�G
=��z��י�Ck�\�
=�1�����RCh��                                    Bxc\t�  "          @�\)�AG���H�p����CZ���AG��33�#�
�{CV@                                     Bxc\��  "          @�(��J�H��=q����{CQ�R�J�H��
=�)���ffCLY�                                    Bxc\�J  �          @�(��L���l(��
�H��G�C��f�L���S�
�.{�Q�C�9�                                    Bxc\��  �          @������^{� ���=qC�|)����B�\�AG��1G�C���                                    Bxc\��  �          @�
=�G��A녿�\)��
=ClG��G��-p��z��	��Ci8R                                    Bxc\�<  T          @�{�C33�QG�������\)Cc
=�C33�AG���\)��z�C`�q                                    Bxc\��  
Z          @��\�N{�W
=���\�s\)CbB��N{�HQ��\��(�C`.                                    Bxc\ۈ  �          @����J=q�W
=��  �q�Cb�J=q�HQ��  ���
C`�3                                    Bxc\�.  �          @�G��Mp��P�׿�\)��\)Ca��Mp��@�׿�{���C_:�                                    Bxc\��  T          @�33�L(��U���G���\)Cb=q�L(��C�
� ���îC_�=                                    Bxc]z  
Z          @��H�E��a녿��
�uG�Cd�
�E��R�\����\)Cb��                                    Bxc]   �          @��xQ��AG�������(�CY�R�xQ��/\)�33����CWE                                    Bxc]$�  �          @�ff��{�1녿���g\)CU����{�"�\���H���
CSE                                    Bxc]3l  T          @�z���z��0  ��ff�iCU����z��!G����H��33CSaH                                    Bxc]B  
�          @�=q���H�'���(�����CT�����H�
=��{��  CR                                    Bxc]P�  
�          @����}p��-p���(�����CVn�}p���Ϳ������CS�=                                    Bxc]_^  
�          @����mp��@  ��ff��z�CZ�3�mp��.{�   ���CXE                                    Bxc]n  �          @�ff�^{�N�R��=q�z�\C^���^{�?\)�����RC\��                                    Bxc]|�  
�          @���a��Dz`\)��Q�C]��a��4z��=q���CZ��                                    Bxc]�P  �          @�z��`  �N�R����MC^�R�`  �A녿�����=qC\�
                                    Bxc]��  
�          @�G��c33�1녿�=q���HCZ
=�c33�   ���R��\)CW#�                                    Bxc]��  �          @��H�u��H��
=����CT=q�u�Q���\���CQ
=                                    Bxc]�B  T          @�\)�w��,�Ϳ�������CV�f�w������H��  CT{                                    Bxc]��  
Z          @�z��\(��R�\�n{�1��C_���\(��Fff��
=��G�C^
                                    Bxc]Ԏ  
�          @�33�fff�W���ff�lQ�C_��fff�HQ��ff���
C\��                                    Bxc]�4  
�          @�33�|���;������t��CX���|���,(����
���
CVJ=                                    Bxc]��  T          @�
=� ���tz�xQ��?�Cl�R� ���hQ��ff����CkQ�                                    Bxc^ �  �          @����P  �P  ��  �C�
C`��P  �C33���R��p�C_+�                                    Bxc^&  "          @����l���7��u�9G�CY���l���,(������p�CW�q                                    Bxc^�  T          @�{��G��*=q�}p��8z�CUW
��G��{�����z�CSxR                                    Bxc^,r  
�          @�Q���z��p��fff�&�\CN�q��z���\��p��d��CL��                                    Bxc^;  �          @����*=q�k�>u@E�Cj)�*=q�j�H������z�Cj\                                    Bxc^I�  T          @��\�g
=����,����C:�=�g
=<#�
�.�R��C3�
                                    Bxc^Xd  
�          @���^�R��33����(�CI�f�^�R���
�{��
CDxR                                    Bxc^g
  �          @����Mp���������CQ�)�Mp���
=�)����HCL                                    Bxc^u�  T          @�\)�`  ���=q����CR�=�`  ���
�Q���CN�                                    Bxc^�V  �          @�z��U���ÿ�\)�ə�CT���U���=q����=qCP��                                    Bxc^��  �          @�\)�C33�{��\)��\)C[�C33��R��p����CX+�                                    Bxc^��  �          @�Q��:=q�:=q����f{C`��:=q�-p����R���RC^��                                    Bxc^�H  T          @�G��Y���
=q��=q����CTc��Y������33���CQ�                                     Bxc^��  "          @����+��L�Ϳn{�H  Ce���+��@�׿�z���=qCdO\                                    Bxc^͔  
�          @�G���(���33��\)�b�\C~G���(���  �s33�BffC}�                                    Bxc^�:  T          @��H>#�
��>�@�
=C��>#�
��ff�L���.�RC��                                    Bxc^��  "          @�ff?�p���p�?s33AE�C�>�?�p�����>�z�@p��C��                                    Bxc^��  "          @��?�  ��=q?h��A@(�C��\?�  ��p�>��@[�C�aH                                    Bxc_,  
�          @���>W
=���>�ff@�  C�g�>W
=��Q�aG��>{C�ff                                    Bxc_�  �          @����������?+�A33C�Lͽ�����G�    ��C�N                                    Bxc_%x  �          @�녽�\)��\)?E�A$��C����\)����=���?��C���                                    Bxc_4  "          @�z�>L����G�?h��A?33C�XR>L����(�>k�@=p�C�P�                                    Bxc_B�  
�          @�>������?�  AP��C�>�����>��R@��\C��R                                    Bxc_Qj  "          @���>��R����?uAJ�RC�3>��R��(�>�\)@k�C��                                    Bxc_`  
�          @�z�>������R?�Q�A{�C�>�����33?�\@�
=C��q                                    Bxc_n�  
(          @��
>�����  ?\(�A5��C�8R>������H>8Q�@ffC�,�                                    Bxc_}\  T          @q녿Q��^�R������  C�XR�Q��P�׿�z���=qC��                                    Bxc_�  "          @y���L���o\)���Ϳ�
=C��\�L���k��.{�$Q�C��
                                    Bxc_��  "          @��\��(���Q�>���@�p�C�����(���Q쾙����=qC���                                    Bxc_�N  
�          @��׾�\)�\)>���@�G�C�H��\)�\)������  C�H                                    Bxc_��  
�          @\)<��|(�>�p�@�33C�/\<��|�;���n{C�/\                                    Bxc_ƚ  T          @���=�Q�����>��@��\C��H=�Q���G��k��O\)C��H                                    Bxc_�@  
�          @z=q>�  �w
=>�(�@ə�C��{>�  �w��B�\�0��C��3                                    Bxc_��  
�          @���?.{�l��?���A�p�C�*=?.{�xQ�?G�A3�C��                                    Bxc_�  "          @��\<#�
�p  ?��RA�G�C�
=<#�
�|(�?aG�AH��C�
=                                    Bxc`2  T          @�  �
=�r�\?�{A�G�C��R�
=�z�H?   @��C��)                                    Bxc`�  �          @��\�G��w
=?�ffAo
=C�T{�G��~�R>�(�@�  C�~�                                    Bxc`~  �          @�\)�\(����R?�G�A���C�:�\(����?z�@���C�k�                                    Bxc`-$  �          @�{�u�}p�?�  A�ffC�(��u��{?���Af�\C��                                    Bxc`;�  
�          @��Ϳ@  �~�R?�
=A�Q�C����@  ��ff?��
AX��C��                                    Bxc`Jp  
(          @���.{�z�H?��
A�G�C���.{����?��Ar=qC�U�                                    Bxc`Y  
Z          @��
��G��l��@�
A�=qC~Ǯ��G��~�R?���A�\)C�\                                    Bxc`g�  	�          @������hQ�@	��A��C}Y�����{�?��A�p�C~�=                                    Bxc`vb  
�          @��������Vff@(�B
��C|G������l��?�\)AϮC}ٚ                                    Bxc`�  
�          @��ÿ�
=�S33@��B	(�Czff��
=�h��?�=qẠ�C|)                                    Bxc`��  �          @�=q�����N�R@(��B{C{� �����g
=@A��C}�\                                    Bxc`�T  �          @�{�O\)�AG�@0��B%��C~�O\)�Z�H@  B=qC�S3                                    Bxc`��  "          @|(��#�
�7
=@,��B-z�C�e�#�
�P��@p�B��C��
                                    Bxc`��  �          @s�
=u�&ff@0  B:p�C�� =u�@��@33B�C��=                                    Bxc`�F  
�          @]p�>��(Q�@��B�C��3>��;�?�Q�A��HC�o\                                    Bxc`��  T          @aG�>W
=��
@5�BW=qC��q>W
=�\)@p�B2\)C�^�                                    Bxc`�  
�          @tz�=�Q��\)@7�BDG�C��=�Q��:=q@�B=qC���                                    Bxc`�8  "          @k���  �z�@7
=BKp�C�箾�  �0  @��B&�C�c�                                    Bxca�  T          @e�=�\)��(�@B�\Bq�
C�.=�\)�(�@.�RBL�C��                                    Bxca�  	�          @aG�?5�z�@!�B8��C�w
?5�,��@�B�C�T{                                    Bxca&*  �          @X��?�  ��@��B3�HC�S3?�  �"�\@ ��B��C��                                     Bxca4�  
�          @P��?��˅@�B(�C�XR?���?��B�\C��H                                    BxcaCv  
�          @]p�?�33��=q@"�\B@ffC�+�?�33��(�@  B%{C�                                      BxcaR  "          @H��?�z`\)@�
B%33C��?�z��Q�?�BC�5�                                    Bxca`�  �          @Dz�?�p��\?�G�B
=C�H�?�p����
?��RA�{C�H                                    Bxcaoh  
Z          @0  ?�z�s33>���A�C���?�z�}p�>\)@s�
C�N                                    Bxca~  "          @5���\?��\�	���H�\C�����\?�{��z��,ffC J=                                    Bxca��  "          @$z῞�R?(���\�_C�{���R?p�׿��K��C�                                    Bxca�Z  �          @(�ÿ��?@  ��ff�+�RC^����?��
��z��  C�                                    Bxca�   T          @*=q�{>�{��  ��(�C+Y��{?
=q���ٙ�C&c�                                    Bxca��  
�          @P�׾��H?�z��>�R�Bᙚ���H?���0  �h��B�G�                                    Bxca�L  �          @aG����?�(��2�\�[CO\���?�z�� ���=B�33                                    Bxca��  
�          @p�?��׿:�H�aG�� Q�C�?��׿0�׾�{�E��C��\                                    Bxca�  �          @j=q@ff��p�?�(�BC���@ff��?�\)Aң�C���                                    Bxca�>  �          @w
=@�R���H@��B��C��@�R��@z�B�\C���                                    Bxcb�  
(          @x��@'��G�@G�A�33C�:�@'��z�?�z�AʸRC�>�                                    Bxcb�  
�          @z�H@0����?��A�\C��
@0���z�?\A�p�C��                                    Bxcb0  
�          @~{@%����?�A�(�C�l�@%��"�\?\A�(�C��
                                    Bxcb-�  "          @�  @,(��G�@	��BffC��f@,(��ff?��
A�z�C�j=                                    Bxcb<|  �          @�(�@8����?��HA�C���@8����H?�=qA�(�C��                                    BxcbK"  
�          @��@8Q���
@�A�RC��H@8Q��'�?�z�A��RC��{                                    BxcbY�  "          @}p�@:�H��\?���A�
=C�` @:�H��
?\A��HC�n                                    Bxcbhn  
�          @}p�@G
=��z�?�(�A�{C��f@G
=��(�?��HA�\)C��                                     Bxcbw  �          @o\)@7
=��ff?�A���C��@7
=� ��?���A�C�k�                                    Bxcb��  �          @a�?c�
�XQ�>�G�@�{C�Y�?c�
�Z=q���	��C�K�                                    Bxcb�`  �          @[�?��
�Dz�?@  AK\)C�G�?��
�J=q>aG�@p��C��                                    Bxcb�  T          @^�R@ff�(�?��
A�C�Y�@ff�'�?Y��Ad(�C�XR                                    Bxcb��  
�          @\(�@G���?Q�A_\)C��@G��!�>���@�G�C��                                    Bxcb�R  �          @aG�@*=q�p�?�RA#
=C�+�@*=q��>u@z=qC���                                    Bxcb��  "          @Z�H?�ff�8Q�<��
>�\)C��)?�ff�5������RC�(�                                    Bxcbݞ  �          @g�?Q��\�Ϳ(���(z�C���?Q��R�\��p���{C�
=                                    Bxcb�D  �          @]p�?fff�L(��L���X��C���?fff�@�׿�=q��p�C�H�                                    Bxcb��  �          @Tz�?z�H�Fff��R�-�C���?z�H�<�Ϳ����
=C�8R                                    Bxcc	�  �          @HQ�?��7����
���C�)?��1G��L���m�C�s3                                    Bxcc6  T          @HQ�>��޸R�
�H�L��C�33>�����(��t�C��R                                    Bxcc&�  
Z          @=q?J=q��
�fff���C�u�?J=q��׿��R���\C�b�                                    Bxcc5�  
Z          @C33��G����R�
=q�<�HC�b���G��˅�{�d=qC���                                    BxccD(  �          @`��@-p���Q�?�=qA��C�7
@-p��ff?:�HA?�C�'�                                    BxccR�  
�          @a�@3�
���?��A��C�� @3�
���H?O\)AV=qC��                                    Bxccat  "          @QG�@,�Ϳ�z�?�\)A���C�>�@,�Ϳ�=q?\(�At��C���                                    Bxccp  �          @J=q@'
=��z�?���A���C��@'
=��=q?O\)An�HC�s3                                    Bxcc~�  
Z          @L��@,�Ϳ�33?z�HA���C�U�@,�Ϳ��?8Q�AP��C�                                      Bxcc�f  "          @3�
@��=q?\(�A�G�C��
@���H?(��A`(�C�J=                                    Bxcc�  "          @B�\@G����H=�\)?�{C��@G�������=q��p�C��                                    Bxcc��  
Z          @7�@���{=u?�C��f@�����=q��=qC���                                    Bxcc�X  �          @K�@�\���#�
�.{C�4{@�\��þ�
=��=qC�xR                                    Bxcc��  T          @C33@����>B�\@dz�C���@���������C��=                                    Bxcc֤  
Z          @>�R@(Q쿳33���
����C��@(Q쿮{���
���
C�XR                                    Bxcc�J  �          @?\)@,�Ϳ�  �������C��)@,�Ϳ�
=����RC�/\                                    Bxcc��  �          @N{@:�H���H�
=�*{C��
@:�H����L���ep�C�                                    Bxcd�  "          @U�@=p���Q�s33��  C��@=p���G������=qC���                                    Bxcd<  T          @S33@4z῜(�������ffC�U�@4z�}p�������C�]q                                    Bxcd�  
�          @Z�H@.{��
=��{����C�  @.{���˅��{C�`                                     Bxcd.�  �          @QG�@+��\�����ffC�@ @+���ff������Q�C��                                    Bxcd=.  �          @P  @*=q���
���
���\C�f@*=q��=q�����G�C��H                                    BxcdK�  T          @L��@#�
����(�����C��)@#�
�c�
��33���C�n                                    BxcdZz  T          @N{@%��xQ��\)��C���@%��+���\�(�C��                                     Bxcdi   �          @H��?�Q��{?=p�Alz�C���?�Q��z�>���@�G�C�\                                    Bxcdw�  
�          @B�\?�=q�Q��G��ffC��=?�=q��׿\(����C��                                     Bxcd�l  
Z          @HQ�?��\�.{�W
=�z�RC���?��\�!G�����Ǚ�C�]q                                    Bxcd�  
�          @4z�@녿�  ?333Aj{C���@녿�>�p�@���C���                                    Bxcd��  "          @,��?��
� ��>�{@�  C���?��
��\����C���                                    Bxcd�^  �          @P  @   �#�
>#�
@4z�C�f@   �"�\�������C��                                    Bxcd�  
�          @R�\@�
�"�\>aG�@q�C��f@�
�"�\��  ����C��=                                    BxcdϪ  �          @N{@
�H�33?!G�A3
=C��H@
�H�Q�>L��@a�C�+�                                    Bxcd�P  �          @R�\@�����u�}p�C�Ф@�������H�Q�C��                                    Bxcd��  �          @Z�H?��#33�L���eC��3?��
=��  ��Q�C��                                     Bxcd��  �          @Tz�?�\)�&ff��33��z�C��?�\)�ff������RC�G�                                    Bxce
B  �          @^{?޸R�*�H��{���C��
?޸R����=q��Q�C�&f                                    Bxce�  �          @[�?�\�%��������C�8R?�\��\�����C��H                                    Bxce'�  �          @fff@��   �h���l  C��)@���\������(�C���                                    Bxce64  �          @g
=@���%��#�
�#�
C�k�@����H�������C�W
                                    BxceD�  �          @u�@3�
�!�>.{@#�
C��@3�
�!G�������G�C�3                                    BxceS�  �          @|��@HQ���?���A���C�� @HQ����?�G�A�=qC��\                                    Bxceb&  �          @�\)@\����\?��A�33C���@\���  ?h��AH(�C�n                                    Bxcep�  �          @�\)@W
=��?�
=A�=qC�(�@W
=���?�ffA���C�n                                    Bxcer  �          @�{@P  ��?У�A�33C���@P  �?�(�A�
=C��                                    Bxce�  �          @�ff@W���?���A�Q�C�q�@W��G�?�ffAh(�C�f                                    Bxce��  �          @��@@  �Q�?��A���C���@@  �%?aG�AG�C���                                    Bxce�d  �          @���@5�%?��\A���C��
@5�1�?E�A/
=C�Ф                                    Bxce�
  �          @���@2�\�8��?�  A_\)C��@2�\�AG�>�G�@�33C�Y�                                    BxceȰ  T          @�(�@Dz��G�?���A�{C�XR@Dz���?�Q�A�  C�g�                                    Bxce�V  T          @��@)���*�H?���A�C�e@)���:=q?��Amp�C�                                      Bxce��  �          @��@/\)�(�@G�AC���@/\)�!�?���A���C���                                    Bxce��  �          @�33@-p��?��RA�=qC��)@-p��*�H?�G�A�
=C���                                    BxcfH  �          @�=q@!���@	��B (�C���@!��,��?�A¸RC��3                                    Bxcf�  
�          @��H@'���@\)B�\C�'�@'��#�
?��A�Q�C��{                                    Bxcf �  �          @�z�@9���#�
?��
AnffC�J=@9���-p�?�@��C�|)                                    Bxcf/:  �          @�33@N�R�8��>�p�@�33C��@N�R�9���B�\�   C��                                    Bxcf=�  �          @���@P���6ff?8Q�A�RC�h�@P���<(�>#�
@C���                                    BxcfL�  �          @�z�@G
=�>{?\(�A4z�C�%@G
=�E�>�=q@c�
C��                                     Bxcf[,  �          @��
@J=q�7
=?h��AA��C��@J=q�>�R>�33@�=qC�W
                                    Bxcfi�  �          @���@HQ��0��?s33ALz�C�H�@HQ��8��>���@��\C��                                     Bxcfxx  �          @��@C�
�%�?.{A=qC��@C�
�*=q>.{@��C�|)                                    Bxcf�  �          @��@L(��+�?��
A]p�C���@L(��5�>��H@�G�C�8R                                    Bxcf��  �          @�=q@XQ���?���A�ffC���@XQ��(�?�\)Ao�C�{                                    Bxcf�j  �          @��@S33�
�H?��HA���C�Y�@S33�p�?�  A�G�C���                                    Bxcf�  �          @���@1G��$z�?��HA��HC��3@1G��9��?�A��C��
                                    Bxcf��  �          @�{@-p��%�?�\)AԸRC�9�@-p��9��?�=qA��RC���                                    Bxcf�\  T          @�  @9���1�?��\A���C�
@9���>{?333A�C��                                    Bxcf�  �          @�33@*�H�<��?s33AW�C�@*�H�E�>�Q�@��C�q�                                    Bxcf��  �          @\)@���@��?h��AR=qC��@���HQ�>���@���C��                                    Bxcf�N  �          @z�H@.{�#�
?�=qA���C�]q@.{�.�R?��AffC�}q                                    Bxcg
�  �          @p��@+���?W
=AW\)C�Ф@+����>\@��RC�3                                    Bxcg�  �          @A녿Y���33��G����Cy�R�Y����z����9  Cu�R                                    Bxcg(@  �          @O\)�:�H�0  ��Q��֏\C
�:�H�=q�����Q�C}
                                    Bxcg6�  �          @Dz�xQ��{�\��=qCx�׿xQ�����p��!p�Cu}q                                    BxcgE�  T          @I��>B�\���ff��C�\)>B�\�녿޸R�!�C���                                    BxcgT2  �          @\��?�p��7
=    <�C��)?�p��333����C��q                                    Bxcgb�  �          @P  >�\)�?\)�u��\)C��>�\)�/\)�����\)C���                                    Bxcgq~  �          @Y��?�=q�<(�?5AEp�C�0�?�=q�AG�=���?�G�C��                                    Bxcg�$  �          @[�?����E�#�
�=p�C��?����AG��(���5G�C�C�                                    Bxcg��  �          @w�@'
=�#33?�=qA�(�C���@'
=�-p�?��A�
C��3                                    Bxcg�p  �          @x��@*=q�\)?�G�A���C�p�@*=q�,��?:�HA.�RC�P�                                    Bxcg�  �          @Vff@��p�?��\A��
C�C�@���H?J=qA[
=C��
                                    Bxcg��  �          @8�ÿ�p��fff���!{CL�=��p���\���R�0�
CBu�                                    Bxcg�b  �          @z=q�:�H?O\)����p�C$s3�:�H?��
���	�CB�                                    Bxcg�  �          @����H��?�(��"�\��RC���H��?��H����(�C
=                                    Bxcg�  �          @�(��L��?E��p����C&k��L��?�G��  ���C��                                    Bxcg�T  �          @�\)�C33���R�7��,�C9Ǯ�C33>�z��8Q��,�
C.��                                    Bxch�  �          @�{�/\)�J=q�C33�;�RCD&f�/\)����I���CC7)                                    Bxch�  �          @��R�(���{�H���6��CYff�(���
=�\���O�RCM                                    Bxch!F  �          @�=q�(Q���H�`  �;p�CX���(Q쿘Q��u��S�CLh�                                    Bxch/�  �          @�=q�!녿���E�6CT�3�!녿u�W
=�L��CHٚ                                    Bxch>�  �          @��R�\)��
=�S�
�R��C=��\)>�\)�Tz��T{C-�{                                    BxchM8  T          @�����H?z��l(��`(�C&p���H?���_\)�Nz�C�                                    Bxch[�  �          @��\�33?���hQ��c�C&���33?���[��QC�R                                    Bxchj�  �          @��\�-p��.{�k��V�C7��-p�?���h���R\)C'�=                                    Bxchy*  �          @�33�+�>�33�mp��W�RC,���+�?�\)�c33�J��CE                                    Bxch��  �          @�
=��=����u��l�
C1� ��?fff�n{�b�Cc�                                    Bxch�v  �          @�G��
=q>�
=�{��r�HC)
�
=q?��R�p���a�C@                                     Bxch�  �          @���� ��?�\)�}p��i�C��� ��@
�H�e��I�\C��                                    Bxch��  T          @�����R?�{��
=�d��C���R?�p��x���K=qCp�                                    Bxch�h  �          @����#33?�������Y�
C@ �#33@
=q�i���>�C�H                                    Bxch�  �          @�z��'
=?�ff�r�\�M\)C0��'
=@��XQ��0
=C
@                                     Bxchߴ  �          @����/\)?�=q�xQ��K=qC�q�/\)@��\���.ffC
                                    Bxch�Z  �          @����!G�?��H�\)�R��C���!G�@!G��a��2�
C�                                    Bxch�   �          @�ff��?��R����op�CT{��@���|���Q33C��                                    Bxci�  �          @��\�J=q�\��\)�)CMٚ�J=q?
=��ff\C0�                                    BxciL  �          @�(��0�׿n{��
=�HCiff�0��=L����=q¢�3C/�=                                    Bxci(�  �          @��R�=p��0�����H8RC_!H�=p�>��
��(� W
C�                                    Bxci7�  �          @��Ϳ�ff������(�CIO\��ff?���
=  C�                                    BxciF>  �          @�(���
=>\�w��C!�ÿ�
=?��H�l(���C
                                    BxciT�  �          @��׿޸R@z��HQ��<��B���޸R@<���#33���B�{                                    Bxcic�  �          @����33@#�
�,(��&�B�����33@E��z����B�=q                                    Bxcir0  �          @vff�\(�@8Q���H�z�B�uÿ\(�@U��p���p�B�\                                    Bxci��  �          @w����\?����H���l��C�{���\@�\�0���D
=B��
                                    Bxci�|  �          @��ÿ��>����l��\)C$����?����aG��y�
C�                                    Bxci�"  T          @��ÿ�
=���
�`���t�\C4���
=?B�\�[��k�C��                                    Bxci��  T          @�=q��p�����}p�ǮCFuÿ�p�>��|���C�
                                    Bxci�n  �          @����Ϳ��\���H��CY{���ͽ�Q���
=\)C7�=                                    Bxci�  �          @�����\)��=q��(�\)C`+���\)�\��=q��CC}q                                    Bxciغ  �          @�(���Q������\�wp�Cd����Q�0����33��CM�                                    Bxci�`  �          @�(���p��33��G��_p�CmW
��p������ff��C^O\                                    Bxci�  �          @��ÿ���R��G��V  Ck�ÿ���G�����}p�C^�                                    Bxcj�  �          @�33��G��)���dz��J�Ct�)��G����
��=q�w�Cj�                                     BxcjR  �          @�����z��\���^�Cq+���녿��R�u��\Cch�                                    Bxcj!�  �          @��\��\)�
=�e��a(�Cr{��\)��  �~{(�Cd#�                                    Bxcj0�  �          @�녿��R��
=�{��o
=CmLͿ��R�}p�����=qCZ��                                    Bxcj?D  �          @����
=� ������q�HCos3��
=���\���RaHC\Ǯ                                    BxcjM�  T          @�ff�aG���{�y���x�\Ct�)�aG��n{��
==qCbY�                                    Bxcj\�  �          @{��(���
=�`  33Cv�ÿ(��
=�n�Rk�C`E                                    Bxcjk6  
�          @c33?���Z�H�(���+�C��?���K�����  C��                                    Bxcjy�  �          @y��?!G��a녿�33���HC�?!G��G
=�
�H�	�C���                                    Bxcj��  �          @�  ?h���e���R����C�3?h���J=q���{C��                                    Bxcj�(  �          @��?����l(���\)��ffC��{?����QG�����{C��
                                    Bxcj��  �          @��?�p��p�׿��R��ffC��?�p��W��z���
=C��                                     Bxcj�t  T          @��?�G����þ��ǮC���?�G��s�
��33���C�g�                                    Bxcj�  �          @�=q@Q���Q�L���G�C��@Q���ff�������C�q�                                    Bxcj��  �          @���@���=q����C�
C�u�@���{�����HC�o\                                    Bxcj�f  T          @�G�@���R����>ffC��@���H�33��(�C��                                    Bxcj�  �          @�33@
=q���ÿ�R���
C�^�@
=q���׿�����
=C�q                                    Bxcj��  �          @���@*�H�w��8Q���C�Q�@*�H�e��{��(�C�S3                                    BxckX  
�          @��H@/\)�|�;W
=�"�\C�Z�@/\)�r�\����Z�\C��=                                    Bxck�  T          @��\@R�\�w
=��Q쿆ffC�5�@R�\�n�R��  �5�C��3                                    Bxck)�  T          @��@HQ��y��?p��A)�C�aH@HQ���Q�#�
��G�C��)                                    Bxck8J  �          @��H@S�
�E�?��A�G�C���@S�
�XQ�?\(�A%G�C�7
                                    BxckF�  �          @�
=@!G��@��@33A��C���@!G��Z=q?��
A�z�C�4{                                    BxckU�  �          @�{@\)�<��@(�B�C��)@\)�\(�?�A���C���                                    Bxckd<  �          @�(�@��@  @=qB�
C�Z�@��_\)?У�A���C�c�                                    Bxckr�  �          @��@Q��G
=@
�HA�33C�5�@Q��b�\?�{A��C���                                    Bxck��  �          @��H?���9��@.{B
=C���?���^{?���A�G�C��)                                    Bxck�.  �          @�=q@
=q�!�@.�RB=qC�G�@
=q�G
=@�\A�(�C�k�                                    Bxck��  �          @�33?����@6ffB4=qC���?���0  @  B=qC�N                                    Bxck�z  �          @���@p��$z�@��Bp�C��H@p��Dz�?ٙ�A��\C�Z�                                    Bxck�   �          @��R@.�R�$z�@+�Bz�C�` @.�R�H��?�p�A��
C�}q                                    Bxck��  �          @�Q�@�  �/\)@��A��C���@�  �J�H?�z�Am��C���                                    Bxck�l  �          @�Q�@��H�@  ?xQ�AG�C���@��H�I��>W
=@G�C�!H                                    Bxck�  T          @�{@�ff�7�?���Ay�C��3@�ff�J�H?Y��A
=C���                                    Bxck��  �          @�@��4z�?ٙ�A��C�q@��I��?uA��C��{                                    Bxcl^  �          @��@�{�'�?�
=A�ffC��@�{�@��?�p�A@  C�P�                                    Bxcl  �          @�(�@�����H@��A��HC���@����7�?�p�Ai�C��\                                    Bxcl"�  �          @���@�G��z�@\)A�=qC�@�G��333?˅A�(�C��{                                    Bxcl1P  �          @���@�����@ffA�ffC��@����,(�?޸RA�z�C�]q                                    Bxcl?�  �          @��
@���#33@
=qA��HC�@���@��?��HAf�\C��q                                    BxclN�  �          @�z�@������@ffA�\)C��@����&ff?�G�Aw�
C���                                    Bxcl]B  �          @�G�@����@
=A��C�` @����333?��HAtz�C�8R                                    Bxclk�  �          @��\@�
=�%?�A��
C�<)@�
=�?\)?�(�AS�
C�Z�                                    Bxclz�  �          @�(�@���\)@
=qA��C�˅@���<��?�(�A}p�C��
                                    Bxcl�4  �          @��@�����@��AŮC��3@����7
=?˅A��C�/\                                    Bxcl��  
�          @��@�����\@�A�  C�Ff@����!G�?���A�
=C��                                     Bxcl��  �          @�z�@�p����R@�RA�=qC�Ff@�p���R?�z�A���C��q                                    Bxcl�&  �          @���@��\��\@��A�G�C��\@��\�!G�?ǮA��C�%                                    Bxcl��  �          @��@��
��p�@
=qA��C�+�@��
�p�?�=qA�{C��3                                    Bxcl�r  �          @�G�@~�R��p�@ffA޸RC���@~�R� ��?�\A��RC��                                    Bxcl�  �          @�(�@vff����@,��B 33C���@vff�%�@�A�ffC�R                                    Bxcl�  �          @��H@�G���G�@J=qB(�C���@�G��G�@*�HA�{C�S3                                    Bxcl�d  �          @�
=@�{���@G�B
�C�H�@�{���@%A�Q�C�(�                                    Bxcm
  �          @���@�ff��@C�
B(�C�Q�@�ff�"�\@   AׅC�l�                                    Bxcm�  �          @���@��\����@\��B�C��=@��\�+�@7
=A��C�\)                                    Bxcm*V  �          @���@����\)@J�HB=qC�@���'�@%�A�
=C��R                                    Bxcm8�  �          @�(�@�(��
�H@I��B
=C�*=@�(��:=q@�RA�G�C�p�                                    BxcmG�  �          @���@�G��ff@HQ�B
�C�J=@�G��5@�RAՙ�C�w
                                    BxcmVH  �          @��R@�  ���R@H��BG�C��)@�  �/\)@ ��A�
=C�Ǯ                                    Bxcmd�  �          @�
=@��׿�p�@G�B��C��)@����.�R@   A�  C��                                    Bxcms�  �          @��
@xQ���R@HQ�BC�s3@xQ��/\)@ ��Aޏ\C�k�                                    Bxcm�:  �          @��@fff���H@G
=B��C���@fff�-p�@\)A�\)C���                                    Bxcm��  �          @�p�@c�
��Q�@U�B ��C�U�@c�
�   @1G�B��C�w
                                    Bxcm��  �          @���@p�׿��@P��B\)C�=q@p���%@+�A�33C��
                                    Bxcm�,  �          @��@mp��У�@S�
B
=C�  @mp����@0��A��C�K�                                    Bxcm��  �          @�
=@dz῜(�@N�RB"G�C���@dz���@2�\B�
C�,�                                    Bxcm�x  �          @��@p�׿���@]p�B"�C�j=@p���z�@<��B��C�(�                                    Bxcm�  �          @��
@tz��z�@c33B{C�˅@tz��;�@7�A���C�<)                                    Bxcm��  �          @�ff@z�H��Q�@j=qB�\C���@z�H�5@@��A��RC��                                    Bxcm�j  T          @�(�@p����\@j=qB"G�C���@p���<��@>�RB 
=C��3                                    Bxcn  �          @��@g����@k�B${C�Z�@g��Fff@=p�A��RC���                                    Bxcn�  T          @��H@l(���@eB�RC���@l(��C�
@7�A�\)C�*=                                    Bxcn#\  �          @���@_\)��\@s33B-
=C���@_\)�?\)@G
=B�C��                                    Bxcn2  �          @��
@g���Q�@tz�B+�C�� @g��9��@I��B�C���                                    Bxcn@�  �          @�(�@e���@y��B/��C�&f@e�7�@P  B
=C���                                    BxcnON  �          @�z�@fff� ��@uB+C�h�@fff�>�R@I��B  C�8R                                    Bxcn]�  �          @�(�@l�����@g�B {C��q@l���Fff@8Q�A���C��                                    Bxcnl�  �          @���@n�R��@j�HB"�C�5�@n�R�B�\@=p�A�z�C�o\                                    Bxcn{@  �          @��@p  ���
@uB+{C�K�@p  �0��@Mp�B
G�C���                                    Bxcn��  �          @���@mp����@u�B*��C�� @mp��7
=@J�HBffC�'�                                    Bxcn��  �          @��R@s33��G�@x��B+�C���@s33�0  @P��B33C��                                    Bxcn�2  �          @�G�@w
=��p�@uB&
=C�j=@w
=�=p�@I��B��C�B�                                    Bxcn��  �          @�=q@s�
��Q�@|��B+�C���@s�
�<��@QG�B�\C�)                                    Bxcn�~  �          @�=q@u��p�@y��B(=qC�aH@u�>�R@L��B�C�q                                    Bxcn�$  
�          @��R@j�H����@z=qB,��C�f@j�H�<��@N{B	G�C��R                                    Bxcn��  �          @��@qG�����@w�B)�\C�W
@qG��<(�@K�B�C�                                      Bxcn�p  �          @��@p�׿�  @}p�B.��C���@p���1�@Tz�B\)C��                                    Bxcn�  �          @�
=@l(���{@}p�B/
=C��H@l(��8��@Q�B{C���                                    Bxco�  �          @�
=@k����@\)B1(�C�H@k��5@U�B��C�.                                    Bxcob  �          @�\)@j=q�޸R@���B3��C�E@j=q�3�
@Y��B�C�AH                                    Bxco+  �          @�G�@g
=�޸R@�B8�
C�R@g
=�6ff@aG�B�HC���                                    Bxco9�  T          @���@aG���G�@�
=B;�HC���@aG��7�@c33B  C�b�                                    BxcoHT  �          @���@W���z�@�p�BF\)C�޸@W��5@qG�B"{C���                                    BxcoV�  �          @��@U�˅@�\)BI�
C�C�@U�2�\@vffB%�C��                                    Bxcoe�  �          @���@Q녿�ff@�Q�BL��C�XR@Q��1G�@x��B(��C��                                    BxcotF  �          @���@c33��\)@�{B8�HC�!H@c33�>�R@^�RB�RC�H                                    Bxco��  T          @�33@W
=��  @�G�BK�C��\@W
=�/\)@{�B(C�l�                                    Bxco��  �          @��
@HQ쿸Q�@��BW��C���@HQ��.�R@�(�B3=qC�q�                                    Bxco�8  �          @�{@?\)���@�\)B`G�C��
@?\)���@�
=B?
=C��H                                    Bxco��  �          @���@�;��@�Q�By\)C���@�Ϳ�@�p�B]C��                                     Bxco��  �          @��@{��ff@�Bq��C���@{�ٙ�@��BWz�C��                                    Bxco�*  �          @�
=@���p�@���B��RC��@���ff@�
=Bm�
C�Ф                                    Bxco��  �          @�@{��@�\)B�
=C�s3@{�У�@�\)Bt\)C��                                     Bxco�v  �          @���?�ff?#�
@��B��fA��\?�ff�\(�@�=qB���C�7
                                    Bxco�  �          @���@Q�=L��@��\B��q?�Q�@Q쿷
=@�(�BqffC��                                    Bxcp�  �          @��@/\)>�p�@�p�Br�R@�z�@/\)��=q@�=qBj=qC�B�                                    Bxcph  �          @�33@
=?333@�(�B��A���@
=�Q�@��B�(�C�Z�                                    Bxcp$  �          @��\?�p�?}p�@�ffB�A�(�?�p���@���B�ǮC��H                                    Bxcp2�  �          @�  =�G�?5@�{B��)B��=�G��h��@��B���C�`                                     BxcpAZ  �          @�=q?��?�33@��HB�(�B�B�?���k�@���B���C���                                    BxcpP   �          @���?(�@�@��\B�B���?(�>�ff@��RB�L�B��                                    Bxcp^�  �          @��R?n{?ٙ�@�33B�G�BuG�?n{>�@��
B�#�@���                                    BxcpmL  �          @���?�ff?�{@�z�B�(�BL\)?�ff<��
@�z�B��H?c�
                                    Bxcp{�  �          @�p�?�{?�Q�@�
=Bvz�B8?�{>\@�=qB��qA8��                                    Bxcp��  �          @���?��þ��
@���B�C��R?��ÿ�Q�@�33B���C�(�                                    Bxcp�>  �          @�
=��  ���@��B���C��=��  �{@���Bo=qC�)                                    Bxcp��  �          @�����R����@��
B��)C_�ᾞ�R�޸R@��B��C���                                    Bxcp��  �          @��\��=q=���@���B�{C�3��=q��=q@��\B�C�AH                                    Bxcp�0  �          @��?��ÿ��@�\)B�k�C��
?����Q�@|��BSz�C���                                    Bxcp��  �          @���@
=��\)@��
B^Q�C�:�@
=�AG�@W�B)��C�|)                                    Bxcp�|  �          @��R@
=�G�@�z�BTC�ff@
=�Z=q@QG�B��C���                                    Bxcp�"  �          @��R@�
��R@�{BX=qC�T{@�
�X��@Tz�BC��                                     Bxcp��  �          @��\@(��:=q@U�B%
=C��@(��q�@33A�z�C�t{                                    Bxcqn  �          @�=q@{�)��@^{B.��C�}q@{�e�@   A�33C�S3                                    Bxcq  �          @�  ?ٙ���@���Bn�\C�H?ٙ��S�
@l(�B3G�C��H                                    Bxcq+�  �          @��R?˅��\)@�  B�\)C��H?˅�0  @��BQG�C�                                    Bxcq:`  �          @���?ٙ���\)@��Bz�C�+�?ٙ��;�@vffBBz�C��                                    BxcqI  �          @�G�?�
=���@�33B��\C���?�
=�>{@xQ�BJ�C��\                                    BxcqW�  �          @��H?^�R��
=@��\B�\)C�*=?^�R�&ff@��Bd�\C�4{                                    BxcqfR  �          @�33?�����
@�z�B^
=C���?����N�R@S33B$�RC��{                                    Bxcqt�  �          @�@G
=�?\)@6ffBC��@G
=�n{?��A�=qC���                                    Bxcq��  �          @���@z��Z�H@B�\Bz�C��@z���{?�A�=qC�}q                                    Bxcq�D  �          @�z�?����P  @\��B)(�C���?�������@�A�
=C��R                                    Bxcq��  �          @��@'
=�%�@b�\B0�C��@'
=�c33@$z�A�ffC�+�                                    Bxcq��  
�          @�@K���R@N�RB��C��@K��W
=@33A�\)C���                                    Bxcq�6  �          @�\)@\���
=q@Q�B�\C��)@\���Dz�@��A߅C�(�                                    Bxcq��  �          @��R@B�\��R@X��B#\)C�j=@B�\�Z=q@��AᙚC��)                                    Bxcqۂ  �          @��@HQ��33@i��B1�C�^�@HQ��E@5�B�C���                                    Bxcq�(  �          @�z�@AG����R@hQ�B4=qC�E@AG��A�@4z�B��C�s3                                    Bxcq��  �          @��@Fff��(�@U�B(C�˅@Fff�:=q@#33A�
=C�ff                                    Bxcrt  �          @�\)��\�0���\)���C�˅��\���@  �b�C+�                                    Bxcr  �          @��H��z��A���p��U�HCqO\��zῴz���z���C\Y�                                    Bxcr$�  �          @��׿�z��&ff��  �_(�Ci��z�xQ����\.CO�                                    Bxcr3f  �          @�����z���
���R�k��CfxR��z��R��{�=CF�                                    BxcrB  �          @���   ������z�W
CP=q�   ?0����ff��C �q                                    BxcrP�  �          @�(�����=q���HffCXB���>�G���\)� C&p�                                    Bxcr_X  �          @�p���������\)�}�CU�f��>�z���p��)C,n                                    Bxcrm�  �          @�{�%��ff��
=�wffCJ#��%?+������|�RC%��                                    Bxcr|�  �          @���AG��33��Q��G�HCYQ��AG��=p���Q��h�HCA�R                                    Bxcr�J  �          @�  �S33������H�I�\CQ�=�S33���
��ff�`�HC9�                                    Bxcr��  �          @���N�R�aG����R�_�HCC=q�N�R?B�\��\)�aG�C&�\                                    Bxcr��  �          @���1G��!G����\�wC@�f�1G�?�{��  �q�C)                                    Bxcr�<  �          @�Q��=p��W
=����l��CC��=p�?\(�����l�\C#�                                     Bxcr��  �          @�=q�G��s33���\�e=qCD���G�?=p����
�g��C&�)                                    BxcrԈ  �          @�33�c33������R�M�CG�R�c33>������
�W\)C/)                                    Bxcr�.  �          @�z������
��\)�Q�CPT{��?���33�C&��                                    Bxcr��  �          @�33�p  ��z����\�C�CH���p  >����G��O�\C1��                                    Bxcs z  �          @�=q�r�\����R�0��CP�H�r�\�(�����J�C=
                                    Bxcs   �          @�=q�(Q�����z��h��CV��(Q�#�
��{W
C48R                                    Bxcs�  �          @�(��L(���R����Jp�CV��L(��\)���R�hz�C=��                                    Bxcs,l  �          @���A��'
=��ff�@{C\���A녿�G�����f��CFn                                    Bxcs;  �          @���L���=p���ff�/�
C^��L�Ϳ���{�Z��CK�H                                    BxcsI�  �          @�33�p  �7��xQ���CYz��p  ��Q�����C�RCI                                      BxcsX^  �          @�z���  �*�H�s33�z�CU�3��  ��G���
=�;\)CE�=                                    Bxcsg  �          @�{��G��@���Dz����CW{��G���=q�x���G�CK�                                    Bxcsu�  �          @�����ff�1G��HQ����HCS�R��ff��=q�w���CG�
                                    Bxcs�P  �          @����n{��R��ff�=ffCR���n{������X33C<Y�                                    Bxcs��  �          @�(��h�ÿ�Q�����F�CP{�h�þaG���
=�\G�C7u�                                    Bxcs��  �          @���e��\)��
=�K�COxR�e��������_C5��                                    Bxcs�B  �          @˅�g������  �O33CK
=�g�>aG���
=�\��C0�{                                    Bxcs��  �          @���j�H�����33�Rp�CG���j�H>������[33C,��                                    Bxcs͎  �          @�
=�j=q��(���(��Q�HCI�
�j=q>�{��=q�]��C.�q                                    Bxcs�4  
�          @�
=�n�R��
=��  �JCL@ �n�R=�Q������Z�
C2�\                                    Bxcs��  T          @�(��qG���33�����B\)CN� �qG��8Q���z��W  C6��                                    Bxcs��  T          @���P���\)�����Ep�CYW
�P�׿:�H���H�g��C@��                                    Bxct&  �          @У��aG���R�����Iz�CTW
�aG���������d��C:z�                                    Bxct�  �          @У��e�����(��B(�CU�H�e��������`�C=n                                    Bxct%r  �          @����Y���Q�����JffCV�q�Y��������i  C<��                                    Bxct4  �          @�Q��r�\�(Q�����2�\CV��r�\�n{���T
=CA                                    BxctB�  �          @����n�R�'
=���7{CW
=�n�R�^�R��G��XffCA�                                    BxctQd  �          @�G��tz��������9�RCT��tz�!G������VC=n                                    Bxct`
  �          @��H�N�R�6ff��  �E{C]s3�N�R�xQ���p��m�CD�                                     Bxctn�  �          @Ӆ�P���1G������FG�C\O\�P�׿aG�����l�RCC#�                                    Bxct}V  �          @�(��\(��0����ff�A(�CZ���\(��c�
���H�f33CB��                                    Bxct��  �          @�z��H���������TffCY�R�H�þ����\�u�\C<��                                    Bxct��  
�          @�(��G
=�Q���33�WQ�CYff�G
=�Ǯ���H�w33C;&f                                    Bxct�H  �          @�z��HQ��{�����TG�CZO\�HQ���H���\�u��C<�                                    Bxct��  �          @�{�@  �/\)��=q�R�C^\)�@  �:�H��{�z
=CA��                                    BxctƔ  �          @�ff�6ff�-p�����W�HC_���6ff�+�����
=CA8R                                    Bxct�:  �          @ָR�1G��'
=�����]�\C_^��1G��
=q�\B�C?                                    Bxct��  �          @ָR�`��� �����H�F�HCW���`�׿z������g
=C=ff                                    Bxct�  �          @�
=�l���ff���H�E  CTu��l�;�G����H�aG�C:�3                                    Bxcu,  �          @�{�HQ�������R�[��CW��HQ�#�
��(��wC6�                                    Bxcu�  �          @��Tz��   ����NQ�CX�3�Tz��\�����o\)C<�q                                    Bxcux  �          @�{�\(��>{��z��<ffC\Ǯ�\(��������
�e\)CE@                                     Bxcu-  �          @�ff�Z�H�,(���=q�E�\CZ.�Z�H�=p���{�iC@0�                                    Bxcu;�  �          @�{�XQ��/\)��=q�E�C[�XQ�G����R�j�HCA�                                    BxcuJj  �          @�ff�^{�'���33�FG�CY��^{�&ff��{�h��C>��                                    BxcuY  
�          @�
=�_\)�333���R�?��CZ�q�_\)�^�R��(��e��CA��                                    Bxcug�  �          @���H���������H�Ch޸�H������G��Tp�CY�=                                    Bxcuv\  �          @љ��QG���Q��|����\Cf���QG��ff���
�O33CW��                                    Bxcu�  �          @����U���33�u�{Cf���U��{��G��JQ�CX��                                    Bxcu��  �          @�\)�i���I�������&G�C\� �i�������
�R�CIG�                                    Bxcu�N  �          @�p��s�
�����1G��Ώ\Cd@ �s�
�@������ ��CZ@                                     Bxcu��  �          @�p��l(�����=q��z�Cg�)�l(��_\)�z=q�Q�C_n                                    Bxcu��  �          @��
�C33�����*�H��(�Cn{�C33�hQ�����'=qCe��                                    Bxcu�@  �          @��H�.�R��(��#33��Cr�.�R�xQ���
=�&�Cj��                                    Bxcu��  �          @�\)�`  �1G�����8��CZQ��`  �h�����H�_�\CB��                                    Bxcu�  �          @�G��dz��3�
��ff�7�
CZ33�dz�p����z��^z�CB�                                    Bxcu�2  �          @�G��Z=q�/\)��33�?�HCZǮ�Z=q�O\)�����f33CA^�                                    Bxcv�  
�          @�=q�^{�Dz������4z�C]���^{��Q���ff�`33CF�f                                    Bxcv~  �          @����H���W
=��(��4�
Cb�3�H�ÿ��H�����g�
CL��                                    Bxcv&$  �          @�  �e��%���Q��<�
CW� �e��.{���
�_z�C>�                                     Bxcv4�  �          @�ff�j�H�*�H���H�533CX��j�H�Tz�����Yp�C@�=                                    BxcvCp  �          @�{�p���&ff�����2��CV��p�׿G�����U\)C?�                                     BxcvR  �          @�\)�s�
�'���G��2
=CV� �s�
�J=q���Tz�C?�                                    Bxcv`�  �          @�=q�s33��R���
�?z�CR^��s33������33�Z
=C8��                                    Bxcvob  �          @����|���z������;CO�3�|�;B�\��ff�S  C6�q                                    Bxcv~  �          @������\��z���
=�9\)CM!H���\�u���H�M=qC4Ǯ                                    Bxcv��  �          @�G���{�У���  �:G�CI=q��{>aG������HQ�C0��                                    Bxcv�T  �          @����(��������9z�CB����(�?.{���\�=��C+0�                                    Bxcv��  �          @��H�^�R�)�����R�<p�CYB��^�R�:�H����a�\C?��                                    Bxcv��  �          @�p��333�x����z��0G�CjE�333��z�����m�HCVQ�                                    Bxcv�F  �          @��:=q�x�����\�-\)Ci0��:=q��
=��{�j  CU�                                     Bxcv��  �          @�z��*�H������\)��Cn��*�H�=q��Q��cQ�C^�                                    Bxcv�  �          @����?\)�q����\�.Q�Cg�H�?\)�������i{CS:�                                    Bxcv�8  �          @�(��9���e�����7�Cg��9��������Q��q
=CPk�                                    Bxcw�  �          @�p��5��_\)���>�\Cf��5���z����
�v�CNxR                                    Bxcw�  �          @��J=q�E�����B�\C`n�J=q��G�����q=qCE��                                    Bxcw*  �          @�ff�U��-p���(��HG�C[0��U��(������nz�C>\)                                    Bxcw-�  �          @�\)�`  �,������CffCY�H�`  ��R��ff�h33C>                                      Bxcw<v  �          @���Q��aG�����.�Cc  �Q녿�ff���H�c��CM5�                                    BxcwK  �          @�ff�Tz��U������5�Ca\�Tzῦff��p��g(�CIn                                    BxcwY�  �          @׮�p  �   ��  �?�RCU���p  ��G���=q�_Q�C:                                    Bxcwhh  �          @׮�s�
�1G�����6\)CX
=�s�
�B�\��  �[
=C?W
                                    Bxcww  T          @�Q��|���.{�����3=qCV���|�Ϳ:�H��{�VG�C>��                                    Bxcw��  �          @���}p��!���p��9Q�CT��}p����H��Q��XQ�C;                                      Bxcw�Z  �          @��H���H�����p��833CRk����H��p����R�T=qC9&f                                    Bxcw�   �          @�=q�x���"�\����<�CU+��x�þ�����\�[�C:�H                                    Bxcw��  	�          @���u������\�A  CTff�u��������
�^��C8޸                                    Bxcw�L  �          @�Q���������H�733CH�q���>�z����
�D��C033                                   Bxcw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxcwݘ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxcw�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxcw��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxcx	�  	�          @���,���dz����
�CG�Ch�H�,�Ϳ����33�~��CNs3                                    Bxcx0  �          @���333�e�����?�HCh��333������G��z�\CNB�                                    Bxcx&�  \          @�G��A��`�����R�;Ce0��A녿�����s33CKٚ                                    Bxcx5|            @ٙ��?\)�^{�����>\)CeJ=�?\)���
���R�u��CK5�                                    BxcxD"  	�          @�=q�J�H�U������>33Cbk��J�H������qp�CG�{                                    BxcxR�  "          @����Dz��U������@  CcT{�Dzῑ����tQ�CHO\                                    Bxcxan  �          @�G��?\)�Y����=q�@�
Cd�
�?\)��
=��\)�v�CI�                                     Bxcxp  T          @��
�Tz��\�����733Cb&f�Tzΰ�
��(��k=qCI#�                                    Bxcx~�  
�          @��
�S33�Z=q���R�9�Ca�q�S33��p������l�CHn                                    Bxcx�`  �          @ۅ�U��\(���p��7
=Ca�f�U���G�����j�
CH�=                                    Bxcx�  
�          @أ��C33�L(����
�D��CbO\�C33�u���R�w(�CEc�                                    Bxcx��  �          @ٙ��R�\�R�\���R�;=qCa  �R�\��{��33�m33CF��                                    Bxcx�R  �          @��H�\(��R�\����7z�C_���\(���\)����h(�CF�                                    Bxcx��  
�          @�33�_\)�S33��(��5��C_Y��_\)���������f=qCE�q                                    Bxcx֞  "          @�(��\���N{����:G�C^�R�\�Ϳ��\��33�iz�CDc�                                    Bxcx�D  "          @��H�vff�J=q���R�-��C[aH�vff������=q�Yp�CC}q                                    Bxcx��  
Z          @�G��l���K���
=�033C\�q�l�Ϳ�=q���H�]�CDQ�                                    Bxcy�  "          @����a��S33��\)�1��C_{�a녿�
=�����bp�CF�                                     Bxcy6  �          @أ��W
=�l(���=q�)�
Cc�3�W
=��������b�CM�                                     Bxcy�  "          @�  �W
=�p������&��Cd8R�W
=��Q���33�`��CN��                                    Bxcy.�  �          @ڏ\�h���e������&ffC`�
�h�ÿ�G����\�[(�CJ�=                                    Bxcy=(  
�          @�G��w
=� ����
=�<�\CU{�w
=��33�����\\)C98R                                    BxcyK�  "          @ٙ���  �\)�����>�CQ:���  ��\)��  �X  C4�q                                    BxcyZt  T          @��H�����
=q��=q�?33CP�����<��
��Q��V��C3�q                                    Bxcyi  
(          @�(��c33�0  ��
=�EG�CY�{�c33����z��j�C;aH                                    Bxcyw�  �          @��
�^�R�>{��(��A{C\���^�R�333��z��kC?Q�                                    Bxcy�f  T          @ۅ�QG��>�R��\)�G
=C^Y��QG��(������sp�C?c�                                    Bxcy�  T          @�z��J�H�E������HffC`!H�J�H�8Q���=q�wG�C@�\                                    Bxcy��  "          @��R�\�Mp���p��A\)C`J=�R�\�c�
�����q�CC�                                    Bxcy�X  T          @�(��X���U���
=�8��C`}q�X�ÿ�=q��(��kQ�CE�
                                    Bxcy��  
�          @�{�W
=�O\)����>�\C_�3�W
=�k���\)�o(�CCff                                    BxcyϤ  T          @�G��J�H�U����H�D�CbaH�J�H�k���
=�x�CD#�                                    Bxcy�J  T          @��P���<�����\�O  C^.�P�׾����G��y�C<:�                                    Bxcy��  
�          @�\)�L(��7
=����VG�C]���L(���z���
=�~��C9@                                     Bxcy��  
�          @����O\)�L(����MffC`��O\)�!G���\)�|
=C?
                                    Bxcz
<  
�          @����H���Tz���p��L�Cb�f�H�ÿB�\�����~CA��                                    Bxcz�  T          @�\�O\)�S33���K\)Ca}q�O\)�:�H�����|
=C@�R                                    Bxcz'�  T          @�=q�x���/\)��\)�D{CW&f�x�þ������
�e�HC8aH                                    Bxcz6.  �          @陚�|(��'������F(�CU�H�|(��#�
��(��e
=C6G�                                    BxczD�  �          @�
=�]p��L�������KG�C^�{�]p��
=�ҏ\�w�
C=��                                    BxczSz  
�          @���^{�B�\��\)�Q�C]B��^{��Q���ff�z�C9��                                    Bxczb   
�          @����n�R�@  ��=q�J\)CZ�{�n�R�\��G��pC9�
                                    Bxczp�  �          @���tz��2�\����MQ�CX!H�tz��������n��C6:�                                    Bxczl  "          @��
�c�
�;�����R�C[}q�c�
�aG���\)�xQ�C7xR                                    Bxcz�  �          @��Z�H�J=q��33�Rz�C^��Z�H����ۅ�}Q�C:�=                                    Bxcz��  �          @��j�H�r�\��=q�:
=Ca���j�H��Q���(��o{CF�                                    Bxcz�^  �          @��_\)�w���G��:�
Cc��_\)���\��z��r�HCH
=                                    Bxcz�  �          @���\���xQ����9�CdaH�\�Ϳ�����G��r33CI�                                    BxczȪ  �          @��Z=q�{���(��7�HCe{�Z=q�����У��rG�CJ)                                    Bxcz�P  T          @��Z�H�k������>�Cc&f�Z�H����љ��u(�CE�                                    Bxcz��  T          @�
=�_\)�Y����p��E\)C`=q�_\)�E���=q�uC@z�                                    Bxcz��  �          @�z��\���C33����M(�C]���\�;�
=�Ϯ�wp�C;�                                    Bxc{B  
Z          @�\)�e�7����\�J  CZ�H�e���
��Q��pG�C9!H                                    Bxc{�  "          @�Q��p  �5������F�CY�p  ������ff�j��C8�)                                    Bxc{ �  �          @�(��h���G
=���
�F(�C\z��h�ÿ������pffC<{                                    Bxc{/4  
�          @��H�Vff�c33����Ap�Cb��Vff�xQ��θR�v�CD�                                    Bxc{=�  T          @�p��W��g������@p�Cc
=�W����\�����v�CDǮ                                    Bxc{L�  T          @�33�&ff��33�����9
=Co.�&ff��\���
=qCVE                                    Bxc{[&  �          @�33�p���G���ff�4��Cq�=�p���p����
��CZǮ                                    Bxc{i�  �          @�� ����  �����,z�Cr.� ���\)�љ��{\)C]�R                                    Bxc{xr  
�          @�� ����(���Q��)z�Cr��� ���
=�ҏ\�y�C_B�                                    Bxc{�  "          @�p��-p���
=��G��+  Cp��-p������G��w��C[�                                    Bxc{��  T          @�R�7���{�����*Q�Cn�=�7��
=q�љ��u  CX��                                    Bxc{�d  "          @�\)�>{������\�+(�Cm5��>{����G��t  CV�                                    Bxc{�
  T          @���8��������  �1
=Cm���8�ÿ�Q���p��y�RCU�{                                    Bxc{��  
Z          @���Dz���������-��Ck���Dz��Q��ҏ\�tffCTaH                                    Bxc{�V  �          @�ff�=p���G�����6�Cks3�=p���
=���
�{=qCQ��                                    Bxc{��  
�          @�\)�:=q�x����33�D33Ci33�:=q��Q��ָR��CJ.                                    Bxc{��  �          @�p��+����R��Q��5Q�Co�+����������CV�                                    Bxc{�H  
�          @��
�!G����\�����<p�Co��!G���z���
=�=CUff                                    Bxc|
�  �          @�z��!G����
��z��;��Cp��!G��ٙ���\)ffCU�R                                    Bxc|�  T          @�p��#33���������5��Cp�H�#33�����ff8RCX��                                    Bxc|(:  
�          @��
�#33��������1�\Cq��#33� �����
��CZ33                                    Bxc|6�  
�          @�(��z�������  �*�HCt���z�����33�~�Ca�                                    Bxc|E�  �          @�{�(���������*G�Cs���(��z���(��|ffC_�H                                    Bxc|T,  �          @��33��p����R�)��Ct���33�
=��=q�}z�Ca��                                    Bxc|b�  �          @��ff��(���\)�*\)CtT{�ff�z��ҏ\�}�C`��                                    Bxc|qx  �          @��H�{������z��&�Cv.�{��R�љ��|ffCd.                                    Bxc|�  "          @���
=q���H���H� (�Cw���
=q�1���z��xQ�Ch&f                                    Bxc|��  	�          @���\��
=����!�RCvL���\�*�H���
�x{CeJ=                                    Bxc|�j  �          @�p��
=��33������Cv:��
=�9�������n�
Cf�3                                    Bxc|�  �          @��H���������{��Cu0�����:=q�ȣ��k{Ce޸                                    Bxc|��  �          @�p����G��vff�z�Cw\��W����H�[��Ck.                                    Bxc|�\  �          @�\)�G�����^�R��G�Cw���G��e��G��R(�Cm��                                    Bxc|�  �          @�\)�33��Q��p  �G�Cy�H�33�XQ���  �^  Cn��                                    Bxc|�  �          @�
=���H���_\)��p�Cz����H�h�����\�U��Cq��                                    Bxc|�N  �          @�Q�����ff�Dz����
C{.����=q��=q�E�HCs�                                     Bxc}�  �          @�  �Q���
=�ff����C{
�Q������G��-Q�Cu5�                                    Bxc}�  �          @�  ��{�����QG����
C{}q��{�g
=���H�S  Cr                                    Bxc}!@  �          @����   ���
�����Cx���   �)���˅�y�
Ch�f                                    Bxc}/�  �          @�
=����������Cx����AG����i��Cl�                                    Bxc}>�  T          @����
�H���
����\)Cw\�
�H�8Q����
�i�Ci�                                    Bxc}M2  �          @����{���z�H�p�Cv�=�{�>�R�����d�
CiW
                                    Bxc}[�  �          @߮������=q��p��G�Cy�����*�H���y  Ck@                                     Bxc}j~  �          @�G������33����p�C�9�����<���Ǯ�z�HCv^�                                    Bxc}y$  �          @ᙚ�\���������C}녿\�6ff��Q��z�RCq�                                    Bxc}��  �          @��Ϳ�=q��33��z��8�CxG���=q��
=���
33Cbs3                                    Bxc}�p  �          @�ff�}p���=q�Q���(�C��{�}p��p����ff�CG�C33                                    Bxc}�  �          @ƸR��33����� �����HC�+���33�z�H��p��CG�C}�3                                    Bxc}��  �          @�ff����
=�P����C�9����o\)��p��X�\C|��                                    Bxc}�b  �          @�zῙ�����\�Vff��33C��Ϳ����dz���{�]�C{k�                                    Bxc}�  �          @�ff�O\)��G���G��{C����O\)�@����{�y(�C~�R                                    Bxc}߮  �          @��H�����
�\)��Cz&f������33�x��Ck��                                    Bxc}�T  T          @��H���R��(��b�\��\C{5ÿ��R������H�s��Cn0�                                    Bxc}��  T          @��þ��R�u�z=q�5�RC��ᾞ�R��  ��{33C�\                                    Bxc~�  �          @�ff<#�
�s�
�]p��(��C��<#�
������8RC�1�                                    Bxc~F  �          @�=q>.{�s33�Vff�%z�C�H�>.{���H��p�C�~�                                    Bxc~(�  �          @��\>aG��vff�j=q�.{C�� >aG���{��\)�RC�Y�                                    Bxc~7�  �          @�\)��(��r�\�e�-{C�˅��(�����(�ǮC�z�                                    Bxc~F8  �          @��    �o\)�`  �,ffC��    ��=q��G�#�C�f                                    Bxc~T�  �          @��ý�Q���  �Fff�\)C�c׽�Q��p���z��xQ�C��3                                    Bxc~c�  �          @��\�u�|���,���	Q�C�:�u�
=����q{C��                                    Bxc~r*  �          @��þ�(��vff�O\)�\)C��3��(���\��3333C��                                    Bxc~��  �          @��׾\)�|(��E���RC�׾\)�(���  ��C�9�                                    Bxc~�v  �          @��
���a��fff�5ffC�@ ����=q��G���C}�f                                    Bxc~�  �          @�=q�.{�mp��n{�2G�C��3�.{�ٙ���
=��Cx8R                                    Bxc~��  �          @�33���
�z=q�|(��4�HC�n���
��\��Q��RC��q                                    Bxc~�h  �          @��R?�33�qG��QG��p�C�4{?�33��Q���33�y��C��                                    Bxc~�  �          @�\)?�\)�z=q�R�\���C�f?�\)�33���}�C�Z�                                    Bxc~ش  �          @�(�?�33��G��O\)�C��=?�33�(���{�t(�C�C�                                    Bxc~�Z  �          @�ff?�
=����<���	�C���?�
=���{�iQ�C��
                                    Bxc~�   �          @�{?�������W
=�\)C�޸?�������p��sG�C�                                    Bxc�  �          @�  ?L���s�
�xQ��3C��
?L�Ϳ�Q���p�#�C���                                    BxcL  �          @�  @2�\�L�;�Q����C���@2�\�,�Ϳ޸R����C���                                    Bxc!�  �          @�33@+��k����
����C��@+��'��@�����C��                                    Bxc0�  �          @�p�@�R�i������C���@�R�{�p  �G�HC��\                                    Bxc?>  �          @���?�p��c33��R��z�C���?�p����e��J{C��                                    BxcM�  �          @��R@��N�R��\���
C�
@녿�\)�^�R�Np�C��=                                    Bxc\�  �          @�Q�@��#�
���p�C�>�@녿��
�K��S�C���                                    Bxck0  �          @�33?ٙ��7
=�ff�33C�Y�?ٙ���G��XQ��`ffC�,�                                    Bxcy�  �          @�{?����B�\�)���
=C�7
?��׿���n�R�{�C�.                                    Bxc�|  �          @�=q?�ff�7
=�*=q�$=qC�R?�ff��\)�i��L�C��
                                    Bxc�"  �          @��?��H�A��{�p�C���?��H���H�U�X=qC�z�                                    Bxc��  �          @n{?�  �333� ���G�C�  ?�  �����C33�a�C��                                    Bxc�n  K          @��\?z����B�\�Jp�C���?z�O\)�s�
�3C��                                    Bxc�  "          @��?n{�=p��-p��$z�C��?n{��Q��p  C�h�                                    BxcѺ  �          @��
?+��K����Q�C��)?+���  �fff�y�HC��f                                    Bxc�`  �          @�?.{�Tz�����33C��f?.{����hQ��t=qC�޸                                    Bxc�  �          @��?У��)���'�� �RC��?У׿�Q��a��q33C��{                                    Bxc��  �          @��@G��-p���R���C�W
@G����Mp��Q�RC�n                                    Bxc�R  �          @�33�#�
�@  �,���'�
C����#�
��(��p��W
C�@                                     Bxc��  �          @��\����5�P���C�C�o\���������{� C�N                                    Bxc�)�  �          @�ff���ÿ����|���{Ck�R����>\���  C ��                                    Bxc�8D  "          @�33�����p��`  �r33Cd�쿧�>�  �r�\L�C(�q                                    Bxc�F�  
�          @�
=@x���P�׿����IG�C�f@x�������R���C�,�                                    Bxc�U�  �          @��\@s33�^�R�����p  C�@s33��R�333���C�l�                                    Bxc�d6  �          @���@L���e�\���C��)@L���!G��>�R��\C��f                                    Bxc�r�  T          @�
=@A��{��G���z�C��
@A���R�s33�0p�C�^�                                    Bxc���  T          @���@8���u��(���z�C�z�@8�����l(��1�C���                                    Bxc��(  T          @�(�@:=q�z=q�333��33C�Q�@:=q�{�����E��C�T{                                    Bxc���  T          @�
=@>{�mp��{�ۙ�C�O\@>{�(��xQ��9�HC���                                    Bxc��t  �          @���@,���^�R�(��ӅC��@,���
=�a��7Q�C��                                    Bxc��  T          @�Q�@tz��c33�W
=�ffC��@tz��Dz��ff����C��
                                    Bxc���  �          @�p�@z�H�hQ�����{C��)@z�H�=p�����C�y�                                    Bxc��f  T          @�Q�@\�����Ϳ�(���z�C��)@\���333�h���Q�C�q�                                    Bxc��  �          @�G�@k��tz�Ǯ��p�C��R@k��,(��HQ��	�
C���                                    Bxc���  �          @�G�@n�R�r�\��=q���\C�H�@n�R�)���HQ��	p�C�H�                                    Bxc�X  T          @�z�@o\)�w
=��\)��  C�\@o\)�,���L���
�
C�R                                    Bxc��  �          @��
@u���z��\)��ffC�e@u��<���U��
=qC�8R                                    Bxc�"�  �          @���@u������=q��G�C��@u�?\)�e��RC�
=                                    Bxc�1J  T          @�ff@z=q���
�G���Q�C��=@z=q�=p��r�\���C�h�                                    Bxc�?�  �          @�\)@~{�����
�H��p�C�o\@~{�4z��x���p�C�U�                                    Bxc�N�  �          @Ǯ@w
=�����
=q��Q�C���@w
=�;��{��G�C�l�                                    Bxc�]<  �          @ə�@z�H���R�ff����C��@z�H�@���y���33C�:�                                    Bxc�k�  �          @��
@x����
=���Q�C��@x���:=q���
�!Q�C��                                     Bxc�z�  
�          @�(�@u���������C�Q�@u��(Q��~�R�"Q�C�                                    Bxc��.  �          @�(�@u��u���\��33C�z�@u��
=�r�\� p�C�*=                                    Bxc���  �          @��@p���p  �����C��@p������r�\�#(�C�|)                                    Bxc��z  T          @�G�@�p��w
=�����(�C�p�@�p�����q���C���                                    Bxc��   �          @�\)@Z=q��  �!G����C�>�@Z=q������\�1\)C�w
                                    Bxc���  T          @��
�B�\�U��g
=�:\)C��
�B�\��ff��\)�3Co�{                                    Bxc��l  �          @��׿�p��a��Z�H�${Cs���p���������}��C^8R                                    Bxc��  �          @�{��k��r�\�'�Cph�����
����{�CX&f                                    Bxc��  �          @��R���z=q���H�.
=Cu+����˅���C]^�                                    Bxc��^  �          @�(�����u����
�9�Cp� ��ÿ��\����fCR�                                     Bxc�  �          @�������_\)�j=q�)�Cp�{������z�����~z�CW�q                                    Bxc��  �          @�Q����E�dz��-ffCj�����������H�x  CNǮ                                    Bxc�*P  �          @�Q��p��AG��x���3�
Cfٚ�p��c�
���H�v�CG�f                                    Bxc�8�  T          @�Q���u��\)�9�HCn�����p���
=G�CO�3                                    Bxc�G�  T          @�ff�z��mp���33�C33Cp�{�zῆff��Q�ǮCN��                                    Bxc�VB  �          @�  �����j�H����H�
Cr������s33���
�CN�                                    Bxc�d�  T          @�����s�
���
�;��Cr  �녿�  ��33�3CS�3                                    Bxc�s�  �          @\���n{��{�;�HCs�ÿ���  ���#�CV(�                                    Bxc��4  T          @��ÿ�{�aG�����H(�Cup���{�}p���
=z�CS��                                    Bxc���  �          @�G��7
=�9�����\�;�Cah��7
=����{�r�\C?@                                     Bxc���  �          @θR����7����H���CW@ ��녿&ff��ff�I=qC=�                                    Bxc��&  "          @�z���=q�R�\��=q���CZ���=q���
��(��J�
CB33                                    Bxc���  �          @��z�H�P������� ��C[���z�H�fff��=q�S��C@�q                                    Bxc��r  �          @ٙ���  �S33�����CY�
��  ��  ���R�H\)CA8R                                    Bxc��  "          @�33�xQ��_\)���R�"�C]�q�xQ쿃�
��=q�X��CB�{                                    Bxc��  "          @˅�5��Tz������8  Ce�{�5��Y�������w��CD�3                                    Bxc��d  �          @�Q�>����>�R����l��C��R>��;������ª=qC��f                                    Bxc�
  �          @��R��(��5�����E{CkW
��(��(����(��CFs3                                    Bxc��  T          @����?\)�A��z�H�*�Ca\)�?\)�\(���(��e�CD�                                    Bxc�#V  �          @�  �@  �A����R�2z�CaQ��@  �:�H��(��l  CA�3                                    Bxc�1�  �          @\�B�\�E��  �1�RCau��B�\�B�\��ff�k��CB\                                    Bxc�@�  T          @����K��E���H�*��C`33�K��Tz�����d�CB��                                    Bxc�OH  �          @�z��XQ��G
=���H�&�C^���XQ�Y������]��CB33                                    Bxc�]�  �          @�  �aG��C�
����&�HC\���aG��E�����[
=C@\)                                    Bxc�l�  �          @ȣ��mp��0����\)�)CX�f�mp���������UffC;J=                                    Bxc�{:  T          @ə��Tz��8����Q��6��C])�Tz�����H�g�C;�R                                    Bxc���  �          @�(��3�
�I����33�Iz�CdL��3�
�����\)33C<ff                                    Bxc���  �          @���5��L����33�H(�Cdu��5���ff��  �fC=�                                    Bxc��,  �          @�33�&ff�H����ff�O�Cfh��&ff��Q����33C;�{                                    Bxc���  �          @�ff�Q��@����{�V{Cg�H�Q�u���u�C9�=                                    Bxc��x  �          @ҏ\�   �J�H��ff�P�Cg�   �\�\ǮC<�3                                    Bxc��  T          @�  �ff�>{�����Y\)Cg���ff�#�
���u�C7�R                                    Bxc���  �          @θR���Fff�����Sz�Ch�f����{��  �{C<
                                    Bxc��j  �          @�G��p��J�H���\�I(�ChB��p������  C@�)                                    Bxc��  �          @����ff�u��G��4��Cn���ff���\����\CPn                                    Bxc��  �          @�z������=q��p��/Q�Cy�����Ϳ�  ��ffCc��                                    Bxc�\  �          @�z�u��\)���<�RC��R�u��ff�ÅǮCn.                                    Bxc�+  ]          @�ff��G���=q����E�C���G���ff��  ffCh&f                                    Bxc�9�  
�          @��Ϳ}p���G���ff�M�C���}p������Ϯ�HCd�f                                    Bxc�HN  �          @�
=�z�H�s�
����X  C�\�z�H�Tz���=q�
C\O\                                    Bxc�V�  �          @�G�������
���=qCu\����?&ff�����C��                                    Bxc�e�  
�          @�=q���\�����
G�Ct\)���\?^�R��� Cff                                    Bxc�t@  "          @��
���
��\���.Ci!H���
?E����#�CG�                                    Bxc���  �          @��R�p�׿�ff�����HCrn�p��?�G���  ��C                                    Bxc���  
Z          @�Q����{��33�}�C]���?z�H��=q��C��                                    Bxc��2  "          @�zῸQ쿗
=�\��C[ff��Q�?���ffC z�                                    Bxc���  �          @�=q���H��{��{�=CU\���H?�Q�����.Cff                                    Bxc��~  �          @�����
���
�����C[�����
?˅��
=��C�                                    Bxc��$  �          @�ff�Ǯ�A���G��rz�Cr�=�Ǯ>#�
��Q��C-�q                                    Bxc���  T          @�\)��=q�/\)��ff�w�CpͿ�=q>�p���G��C&�=                                    Bxc��p  
�          @�{��33�#�
���  CqW
��33?���Ϯ�fC�{                                    Bxc��  
�          @�\)��=q�xQ��ȣ��fC^  ��=q@����.B�q                                    Bxc��  �          @�zῙ��������.CL�)����@z������=B�u�                                    Bxc�b  
�          @\��\?333��  £�=B�B���\@\(���
=�\
=B��                                    Bxc�$  T          @�Q�s33>���(�z�C��s33@L(����R�`�\B�{                                    Bxc�2�  T          @����{�#�
��{#�C4=q��{@*�H�����lB��                                    Bxc�AT  T          @�{�h��>�G�����B�CG��h��@@  ��{�`��B��                                    Bxc�O�  T          @�=q���?Q���\) k�B�����@S33��ff�T��B�L�                                    Bxc�^�  �          @�Q�
=q>�  ��(�§\)C�{�
=q@0����(��k
=B�#�                                    Bxc�mF  �          @������{��*�H��RC{�����p���ff�iffCn��                                    Bxc�{�  T          @��\�����p���  ��33C~�쿫��h���h���,�Cy�{                                    Bxc���  T          @��
��z���
=�����\C}ff��z��J=q���H�G�Cv                                    Bxc��8  �          @�=q��ff�^{�O\)�!Cu�f��ff�����  (�C`��                                    Bxc���  
�          @����У��q��:�H�G�Cv�R�У׿��H��33�n��CfW
                                    Bxc���  
�          @��\�Ǯ�s�
�>{�(�Cw�3�Ǯ���H����q�\CgxR                                    Bxc��*  �          @�33�Ǯ�xQ��:=q�=qCx
=�Ǯ�33��z��nQ�Ch��                                    Bxc���  �          @�{��{�c33�0  �p�Cx�ÿ�{��=q���H�sQ�CiY�                                    Bxc��v  "          @�\)��  �/\)�i���M�\Cy�3��  �0����Q���CV�=                                    Bxc��  
�          @�Q��33�[��G
=�  CtW
��33�Ǯ��33�z  C_k�                                    Bxc���  T          @�=q�G��n�R�1G����Cq��G���p���ff�`{C`h�                                    Bxc�h  
�          @�����
�;��~{�H�HCrp����
�5��(��HCL�R                                    Bxc�  T          @�ff��=q�:�H����WCu�)��=q�   �����fCH�)                                    Bxc�+�  �          @�G�����9�������[��Cu�ÿ����
=���� CF�                                    Bxc�:Z  T          @�zΰ�
�C33���H�XCw5ÿ��
����\)��CJL�                                    Bxc�I   �          @�\)��\)�@�������M��Cq�3��\)�!G���{��CIY�                                    Bxc�W�  T          @�33����p  �x���1��C{Ϳ����(���ffǮCd��                                    Bxc�fL  
�          @���333�L(���
=�P�HC��f�333�Q���
=�\Ceu�                                    Bxc�t�  T          @���{��=q�Z=q��
C~� ��{��p����{Cp��                                    Bxc���  T          @�ff��
=��  �C33�
��C~k���
=�33��{�t\)Cr�\                                    Bxc��>  
�          @�\)�
=q��p���  �U��C�P��
=q�L���6ff�$�C�9�                                    Bxc���  T          @�G��u�9��@n�RBP��C�ff�u��=q?�A��
C��R                                    Bxc���  T          @�����G�����@/\)B=qC��q��G����R>�@�Q�C�y�                                    Bxc��0  
�          @��Ϳ������?���Am�Cs3�����33���H  C��                                    Bxc���  T          @����
=��z��/\)��C�n��
=�333����i�C��                                     Bxc��|  �          @��5���H�B�\��\)C�<)�5�QG���ff�d�C���                                    Bxc��"  T          @ə��n{��p��E����C�)�n{�Tz������c(�C~L�                                    Bxc���  �          @���z���=q�?\)���
C�޸�z��QG������d��C���                                    Bxc�n  �          @�녿�����Q��b�\���C��=�����!G����
�{Cw                                    Bxc�  T          @����Q������=q�633C{LͿ�Q��=q������CcǮ                                    Bxc�$�  �          @ƸR�����
=���\�1�\C|�ÿ����
=����qCg��                                    Bxc�3`  T          @ƸR�����G������.(�C|�׿����G���G�#�Ch�q                                    Bxc�B  T          @�
=�����33��G��.CQ쿑녿�ff���\�)Cm��                                    Bxc�P�  T          @��O\)�����(��7��C��f�O\)�ٙ����Ctk�                                    Bxc�_R  
�          @˅�xQ����H�����6
=C��)�xQ��Q���G�W
Cp8R                                    Bxc�m�  T          @�33��{��=q��\)�A33C~�\��{������33
=Cf�3                                    Bxc�|�  �          @�\)��ff��  �����BG�C}���ff��\)��ffk�Cb��                                    Bxc��D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxc���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxc���  �          @��
�W
=�`�����H�h��C�N�W
=��{�ٙ�¤�HCJ�                                    Bxc��6  	W          @�(��Q�������=q�'��C�Z�Q��������Cz�                                    Bxc���  
�          @�녿�  ��G�����!  C�����  � ����Q�  CxY�                                    Bxc�Ԃ  T          @�p������G������!Q�C�*=����'
=��33ǮCvY�                                    Bxc��(  �          @��ÿ�\)��z���(��6��C��R��\)���R��p��RCp�\                                    Bxc���  �          @�p��z�H�L�����H�x�C}Ϳz�H>.{��33¤�C)�                                    Bxc� t  �          @�33���H�6ff���H�|=qCw{���H>�Q��ָRC#(�                                    Bxc�  �          @�p���ff�%����z�Cs0���ff?Tz���  #�C�{                                    Bxc��  �          @�33��33�(���Q�B�CpB���33?k������3C��                                    Bxc�,f  �          @�=q��=q��
�ə�.Chz��=q?�z������=C�                                     Bxc�;  �          @��H��(���p���p��\Ce�Ϳ�(�?�����  B�C��                                    Bxc�I�  
�          @�{�%���R��  �%��Cq:��%��z���p��}�\CXh�                                    Bxc�XX  �          @�z��@��������\)�*
=ClY��@�׿�������y33CP�                                     Bxc�f�  �          @�
=�:�H��\)���
�$�CnO\�:�H��\)�����x{CT�)                                    Bxc�u�  �          @�
=��\)�3�
�ᙚ(�Ct  ��\)?Y�����
C(�                                    Bxc��J  "          @��Ϳ�z��.�R��  �\CrǮ��z�?c�
��R��C��                                    Bxc���  �          @�{��ff�1��ᙚCt녿�ff?aG�����p�C��                                    Bxc���  
�          @�  ��ff�)�����G�Cs녿�ff?�ff��=q(�C�                                    Bxc��<  T          @�\)���H��(����HCe�f���H?��H�陚  B��=                                    Bxc���  �          @�(����\�@  ��R�
CR}q���\@/\)�ָR� B���                                    Bxc�͈  �          @�\��=q������=qffCl��=q@�\��Rz�B��
                                    Bxc��.  �          @����\�Q�����CZ�=���\@0  ��(���B�                                    Bxc���  T          @��ÿ�녿Y����33�CST{���@.{��(��B�Q�                                    Bxc��z  �          @�녿Q녾k���\)¦CCff�Q�@E��Q��{�RB��)                                    Bxc�   �          @ڏ\�W
=�#�
����¥��C6aH�W
=@J�H����u\)B��
                                    Bxc��  T          @�
=�z�>�
=��p�§ǮCn�z�@`����{�g��B�Ǯ                                    Bxc�%l  
�          @�(���?(���=q¦�C�׿�@h����Q��a=qB�W
                                    Bxc�4  T          @��
��{?E����¥k�B�.��{@qG���p��\ffB�aH                                    Bxc�B�  T          @�녾��?xQ���\)¢�)B�Q���@z=q��Q��U�\B�Q�                                    Bxc�Q^  
�          @�G�����?J=q��\)¤��B��)����@p  ���H�[G�B��                                    Bxc�`  �          @θR�u��G���  aHCm�\�u?�\)������B��f                                    Bxc�n�  "          @ҏ\��\�y������Ez�Cu���\����Ǯ(�CS��                                    Bxc�}P  "          @��ÿ��������\�~p�Ck�����?#�
�ȣ��3C��                                    Bxc���  "          @Ӆ����������p�B�Cgٚ����?��\�����C
�                                     Bxc���  
�          @����  �8Q���\)�
CQ�q��  @����
��B�R                                    Bxc��B  
�          @ə����׿}p����HQ�C]쿐��@   ���H{B�\                                    Bxc���  ^          @��H��?�ff��� �RB�� ��@tz���{�Q(�B���                                    Bxc�Ǝ  J          @��H>�{@����ff�fB�  >�{@�{����(�B��R                                    Bxc��4  "          @�  ?�p�@'���ff�t�\BrQ�?�p�@��
�e��p�B�L�                                    Bxc���  	�          @��H��@1G���\)�sp�B�#׽�@�G��E�G�B���                                    