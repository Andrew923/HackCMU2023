CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230217000000_e20230217235959_p20230218021652_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-18T02:16:52.165Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-17T00:00:00.000Z   time_coverage_end         2023-02-17T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxeʺ�  
�          Ad����33A<z�\��\)B����33A0  @��A��B��                                    Bxe��f  
(          A]����
A4���6ff�>�\B�=q���
A3�
@EAO33B��\                                    Bxe��  	�          Ab{��A>�H��{����B��
��A.�H@���A��HB�k�                                    Bxe��  �          Af=q���RA?33<�>\)B�W
���RA&=q@�p�A\B��                                    Bxe��X  |          Ab�R�\)A/
=�:=q�>ffB�ff�\)A/\)@6ffA:{B�Q�                                    Bxe��  T          A]�����A)G���Q����B�������A4��?�  @�\)B�33                                    Bxe��            A\����A$�������z�B�u���A8��?�\@ffB�W
                                    Bxe�!J  �          Ab�\�=qA/�
�7��=�B���=qA/�@9��A?�
B�#�                                    Bxe�/�  
�          AaG���G�A3�
�6ff�<Q�B�z���G�A2�H@B�\AIp�B��R                                    Bxe�>�  T          A^{��\)A-�QG��ZffB�����\)A1�@\)A%G�B���                                    Bxe�M<  T          A_\)��\A,(����R��
=B����\A<��?u@}p�B��                                    Bxe�[�  
�          A^=q��  A"�R��Q�����B��{��  A733>��?�(�B�.                                    Bxe�j�  :          A]G����Ap������C����A)녾��
��=qC�f                                    Bxe�y.  
�          A_\)�
ffA�R��=q��Q�C���
ffA/
=�k��n{C Q�                                    Bxeˇ�  
t          A^�R�A�������͙�C�3�A0�ÿ.{�3�
B�.                                    Bxe˖z  
�          Ad  ���HA/
=���R��ffB�=q���HA=p�?�G�@��\B���                                    Bxe˥   
�          A\����
Az����
����C����
A&ff�
=�{C�{                                    Bxe˳�  
�          AZff�
=A
ff��
=���
C�q�
=A!�����R���CL�                                    Bxe��l  
Z          A_33�G�A,���g
=�o�
B��\�G�A3
=@	��A��B���                                    Bxe��  �          Ab�R��z�A@Q����{B�ff��z�A:�H@qG�AvffB��H                                    Bxe�߸  T          Ag33��AL(��W
=�Q�B�G���A3�@�=qA�p�B�                                    Bxe��^  "          Ae�����AP��@G�A\)B�L�����A$Q�A�BB�\)                                    Bxe��  �          Ac\)��  AEG�@��\A�33B����  AQ�A'�B@�B�#�                                    Bxe��  @          Aap���\)AI@(��A/
=B���\)A33A�B(�B왚                                    Bxe�P  
�          A_��B�\AW\)?Y��@a�B�z��B�\A5�@�\B   B��                                    Bxe�(�  
�          A^�\�ӅA�\@��A�B��R�Ӆ@�p�A/�BTffCJ=                                    Bxe�7�  
�          A`z��dz�AH��@���A��\Bӽq�dz�A�A((�BC33B�{                                    Bxe�FB  "          A_����
AQ�@(�A ��B��
���
A$  A��BB�                                    Bxe�T�  �          A^�R��{AO33@ ��A&�HB��H��{A!�A(�B��B�.                                    Bxe�c�  "          Aa���h��AXQ�@n{Au�B��R�h��A�A��B2  B�u�                                    Bxe�r4  �          Aep�>�=qA_
=@.{A0��B���>�=qA-p�A�HB ��B�.                                    Bxè�  �          Ab�\@���A!G�AB�HB�k�@���@�
=AJffB�{B@
=                                    Bxȅ�  
�          Ai?L��AI��@�p�Aי�B��q?L��@��
A<  B`33B�ff                                    Bxe̞&  T          Al  ���AX��@�G�A�  B�����A�
A3�BH��B�\                                    Bxe̬�  �          An=q?�G�AF�H@���A��B�k�?�G�@ᙚAM�Bs�\B��R                                    Bxe̻r  T          Ap��@"�\AK�
@�RA��HB�z�@"�\@�  AK�Bhp�B���                                    Bxe��  "          Aq�@���A>{Ap�B�
B�=q@���@�{AM��Bi��B[z�                                    Bxe�ؾ  
�          Aqp�@�\)A6ffA
�\B33B���@�\)@�  AQ�Bq�RBJG�                                    Bxe��d  
�          AmG�?�\)A8��Az�B
=B���?�\)@�ffAX(�B���B�.                                    Bxe��
  
�          Am�@�AE@��A�\)B��@�@���ALQ�Bp{B��                                     Bxe��  �          Ak
=?k�AS�
@ƸRAȏ\B�
=?k�A��A=BX�B��3                                    Bxe�V  
Z          Ak�?�z�A=�A	G�BffB��?�z�@ƸRATQ�B�L�B��                                    Bxe�!�  
�          Al(�?�33AN�H@��A��B���?�33A   AC�Ba�B�(�                                    Bxe�0�  "          Aip�?�\)AT  @�{A��\B�L�?�\)A33A9�BSB�G�                                    Bxe�?H  �          AhQ�>8Q�AMG�@�A�{B�k�>8Q�@�\)A@��Bb{B�p�                                    Bxe�M�  �          Af�\��AF=q@��A�33B����@���A?\)BeffB�k�                                    Bxe�\�  
�          Ak��Q�AB�H@��BffB���Q�@�33ALz�BqB�\)                                    Bxe�k:  T          Ai��8��A8(�A{B�
B�.�8��@���ANffBz�\B�Q�                                    Bxe�y�  �          Ag�
�I��A-�A��B�BԀ �I��@��HARffB�  B�                                    Bxe͈�  �          Ac�
��Q�A0��A
=B�RB��쿸Q�@�
=AO33B���B�p�                                    Bxe͗,  "          Ab�H���
A?�@��A���B�8R���
@���ADz�Bq
=B�
=                                    Bxeͥ�  T          Ab=q��
=A8��@�
=B�HBď\��
=@�z�AE��BwffBр                                     Bxeʹx  �          AZ{��AFff@�z�A���B��쿕A�
A$Q�BH��BøR                                    Bxe��  �          AX��� ��APz�@�\ABŏ\� ��A$��A��BG�B��                                    Bxe���  �          AZ�H�A�AQG��p������B���A�A>�\@�\)A�ffBЏ\                                    Bxe��j  
�          AV�\�O\)AB�\@^�RA~�HB����O\)A�A�\B2�RB�aH                                    Bxe��  
�          AW�@L��A�
A ��BBB��@L��@�AL��B�aHB�                                    Bxe���  �          AW\)@\(�@��A%��BHB�@\(�?���AN{B��HA�ff                                    Bxe�\  
�          AQ�@J=qA�A�BA{B�B�@J=q@��AF�\B��=B                                    Bxe�  T          AF{@+�A�@�z�B\)B�
=@+�@���A4z�B���Bd��                                    Bxe�)�  �          A8��@�=qA (�@�p�B�Bv��@�=q@l��A�RBp��B"33                                    Bxe�8N  
�          A5��@�{@��H@�A��HB>=q@�{@���@��B$BG�                                    Bxe�F�  
�          A4��@�ffA��@UA��HBf��@�ff@�=q@��B!��B?��                                    Bxe�U�  �          A3
=@J�HA�@�=qA���B�� @J�H@���A�Ba=qBl33                                    Bxe�d@  h          A0��@j=qA�@�(�A�  B�8R@j=q@�33Az�B]G�BS�R                                    Bxe�r�  
�          A2{@�ff@���@�ffA�z�B]�\@�ff@\)AffBP�B(�                                    Bxe΁�  "          A1�@7
=A�
@k�A��B�
=@7
=@�Q�A Q�B:�\B��{                                    Bxeΐ2  �          A,(�?��Ap�@��AǮB�
=?��@\Ap�BSB�Ǯ                                    BxeΞ�  
�          A'�
?��A  @�B
�\B�?��@�(�AQ�B{(�B�aH                                    Bxeέ~  ;          A,  �G�A  @�33A��B���G�@�33A(�BL�RB�p�                                    Bxeμ$  �          A.{�
=qA!��@Dz�A�Q�B�(��
=q@��@��B1{BԊ=                                    Bxe���  
�          A.{��A)G�?��R@�z�B���A	G�@˅B�B�8R                                    Bxe��p  �          A-��
A&{?�@��HB����
A33@�ffB33BҞ�                                    Bxe��  �          A+����A#�?�\)AG�B˨����A\)@ə�BffB�=q                                    Bxe���  �          A+��VffA ��?�Q�@�B���VffA(�@��HB�
B�.                                    Bxe�b  �          A+
=���A�H�k����RB������A33@s33A��RB�k�                                    Bxe�  
�          A'\)��=qAff��  �33B�aH��=qA�@p�A\  B��=                                    Bxe�"�  
�          A%��p�A (��(��C�
B�33��p�Ap�?��A'�B���                                    Bxe�1T  �          A$����\)@�(���
=�-C T{��\)@�33@33A9�C p�                                    Bxe�?�  	          A%���G�A33��  ���B��
��G�Aff@,��Aq��B���                                    Bxe�N�  
�          A#�
���A Q��
=q�C\)B����Ap�?�z�A,��B�W
                                    Bxe�]F  
�          A"�H��Q�A�R������B�W
��Q�@�
=@(�A^�RB��3                                    Bxe�k�  	          A"ff��G�@���^{��G�B�z���G�A
�H?:�H@�B�G�                                    Bxe�z�  �          A!G����\A�H�<������B������\AG�?��@�
=B���                                    Bxeω8  "          A �����@�33�8�����RB�����Az�?�@أ�B�G�                                    Bxeϗ�  	          A�\��=q@��H�7
=��(�B�=q��=qA(�?�Q�@�B��                                    BxeϦ�  
s          AG�����@��
�e���p�B��
����A�\>�G�@%B�                                    Bxeϵ*  T          AG����HA Q��U��
=B�\���HA
ff?W
=@�p�B��                                    Bxe���  T          A�
��  @�p��P  ����B���  A��?@  @���B��                                    Bxe��v  T          A����@�\)��ff��z�C� ����@��ÿ�G���ffC�H                                    Bxe��  
�          A(���(�@{��p����
C ޸��(�@�(��<����\)Cp�                                    Bxe���  T          A�����@ʏ\�q���(�C�=���@���Q����B�\                                    Bxe��h  �          A(���z�@�Q��vff�ģ�C���z�@�\���4z�B��                                    Bxe�  
�          A���p�@�(����
����CG���p�@�\)���
���B�=q                                    Bxe��  �          A  ���
@ʏ\��z�����B�G����
@�(�������B�Ǯ                                    Bxe�*Z  �          A����@�ff���H���
C=q���@�녿�����
B��=                                    Bxe�9   T          A�H���@�=q�����=qC.���@�Q��p��.�HB�ff                                    Bxe�G�  
�          A����@����,(���{C	ff����@˅>���@
=C��                                    Bxe�VL  
�          Ap����H@�\)���
�C�)���H@��� ���Lz�B��                                    Bxe�d�  
�          A����@��\��p���z�C5����@��H��33�{B��                                    Bxe�s�  �          Ap����@��
����ffCY����@�(���
=�H  B���                                    BxeЂ>  T          A����\@z�H��\)��RC����\@���2�\���HC��                                    BxeА�  �          A����{@��H��(���RC	  ��{@�G��   ��z�B�                                    BxeП�  	          A������@�z��e����B������@񙚽��
��B��3                                    BxeЮ0  
�          A=q����@�G���������C
����@��H���
��B���                                    Bxeм�  �          A������@�p���(���Q�CQ�����@�{���\��G�C �{                                    Bxe��|  T          A���ff@��\��Q�� ��C
��ff@�������d(�C��                                    Bxe��"  	          A�Å@�������C{�Å@�Q�����t��CT{                                    Bxe���  	          A���33@�{�\������C�R��33@�ff�&ff��p�C��                                    Bxe��n  
�          A����Q�@s�
�����G�C����Q�@��H�p����RC
E                                    Bxe�  
�          AQ���p�@u���p��  C
��p�@�G��3�
���C�H                                    Bxe��  
�          A�\��G�@Q����H�0Q�C� ��G�@����=q��(�C�                                    Bxe�#`  T          A���Q�?������G��C$�
��Q�@��\���H�Q�C��                                    Bxe�2  
�          A ������?�{��{�V��C#E����@�{���\�{C	=q                                    Bxe�@�  "          @��R�g
=�G���
=�t�
C@��g
=@=q����]��C5�                                    Bxe�OR  �          @������U�߮�v
=Cq���>�������C+�3                                    Bxe�]�  
�          @����G
=�0�������O{C]�
�G
=���
���R�y�RC4G�                                    Bxe�l�  �          @�(����H?s33�����1
=C)�����H@Q���  ��Cs3                                    Bxe�{D  �          @�33��>���Q��/�HC/(���@5����R�33C}q                                    Bxeщ�  �          @����?�(���\)��\C"�����@l���Z=q��\)C�
                                    Bxeј�  "          @�{���\@
=�����/�
Cn���\@���o\)��z�C��                                    Bxeѧ6  �          @����G�?�Q����7��C$���G�@qG������C:�                                    Bxeѵ�  �          @���=q?�����{�;C!�3��=q@z=q��33�C�R                                    Bxe�Ă  
�          @�\)��Q�@3�
����2p�C���Q�@����W����HC�)                                    Bxe��(  �          @�ff���@+���=q�8z�Cs3���@�=q�`������C(�                                    Bxe���  T          @���Q�@qG���ff�z�C&f��Q�@�(���R��33C�R                                    Bxe��t  �          @�����@8Q����� �HC{����@�
=�9����(�C}q                                    Bxe��  �          @�ff���@�H��
=�.��CO\���@�p��S�
�ۅC5�                                    Bxe��  
�          @�  ��ff?�
=����4C����ff@����j�H����C5�                                    Bxe�f  
Z          @������\@ ����ff�8�CO\���\@���n{��  C	                                    Bxe�+  �          @�(���Q�@$z������ ffC����Q�@���8Q�����C	z�                                    Bxe�9�  "          @����O\)�L�����
�x{C7���O\)@&ff��G��Tz�CE                                    Bxe�HX  �          @�{�g
=>����z��i��C/��g
=@9����(��?��C8R                                    Bxe�V�  
�          @�������?�(����\�Q�C#������@Q��L(���(�C�                                    Bxe�e�  T          @�33��\)@��Vff���C�3��\)@fff��
=����C��                                    Bxe�tJ  "          @ۅ��{@33�s33�=qC&f��{@qG��ff����C�                                    Bxe҂�  �          @�33��(�@\)����(�Cff��(�@z=q�1G���
=CL�                                    Bxeґ�  
�          @�Q���=q@,���aG���{C�q��=q@�  ������C�                                     BxeҠ<  
�          @�\)��=q@Mp��N�R����C���=q@�=q��Q��NffC�{                                    BxeҮ�  
�          @Ӆ���@@  �s33��
C� ���@�z���\��
=C)                                    Bxeҽ�  
s          @�{��G�@
=������HC���G�@{��"�\��z�C	��                                    Bxe��.  ;          @��
���@33�z�H���C�q���@tz��p����C�3                                    Bxe���  T          @θR���?Q��vff�z�C*����@p��Dz���p�CT{                                    Bxe��z  "          @�p����?�{�w
=���C%h����@<���5���
C�H                                    Bxe��   
�          @�p����\>�(��}p��33C/^����\@
=q�Vff��C��                                    Bxe��  T          @�����\)��ff�(33C5����?�\)��G���
C@                                     Bxe�l  T          @�����zᾮ{���R�
=C7��z�?�=q�z�H�=qC"�f                                    Bxe�$  �          @љ����>8Q���
=�$�C1�)���@33�l���  C��                                    Bxe�2�  �          @ƸR�n�R@qG��AG���C���n�R@�
=���\�{C E                                    Bxe�A^  "          @�ff�tz�@\���>{��(�C	�H�tz�@����\)�.=qCٚ                                    Bxe�P  �          @��H��p�@��333��
=C����p�@L�Ϳ���|Q�C!H                                    Bxe�^�  �          @ƸR��33�ff�b�\�{CL���33������33�&C7�                                    Bxe�mP  
�          @Ǯ��=q?���{���\C&���=q@1G��AG����C��                                    Bxe�{�  T          @����G�?�p���G��8  C���G�@k��a���Cn                                    Bxeӊ�  �          @Ӆ��@ ������%�
C��@���0  ���
C#�                                    BxeәB  T          @�
=�1�@�(��u�{B����1�@���?�z�A\z�B���                                    Bxeӧ�  T          @˅�G
=@K�����1�RCc��G
=@���� �����B��                                    BxeӶ�  "          @�  �\��@^{�����$��Cٚ�\��@�  �33��(�B�33                                    Bxe��4  
�          @�z��a�@�Q�5��33B�W
�a�@��?���Al��B��\                                    Bxe���  
�          @љ��~{@���
=����C���~{@�{>B�\?ٙ�B���                                    Bxe��  �          @����y��@�����
��  C&f�y��@�ff>�  @�RB���                                    Bxe��&  T          @�
=��(�@�
=�	�����C���(�@�����
�.{C�R                                    Bxe���  �          @�{����@o\)��33�J{CL�����@~�R>\@[�C�)                                    Bxe�r  T          @��H���R@<(������(�C�����R@fff�0�����C�\                                    Bxe�  "          @�
=��ff?�\)�*�H�Ə\C!
��ff@6ff��=q�f{Cn                                    Bxe�+�  T          @ə����@Vff�%���z�Cp����@�z�Y����Ck�                                    Bxe�:d  
�          @˅?L��@��R����{B���?L��@��@   Aď\B�{                                    Bxe�I
  "          @�녿0��@�z῏\)�*�RB�aH�0��@�G�?���Ap��B���                                    Bxe�W�  "          @ƸR���@�  ��  ��  Bͣ׿��@�?p��A  B��H                                    Bxe�fV  �          @ə��!G�@�{�˅�o
=B�33�!G�@�G�?�33A+�
B�                                      Bxe�t�  �          @��
�5@e����\�-C aH�5@�����z�B��                                    Bxeԃ�  T          @����n�R?��H����K�C�
�n�R@\���r�\�C	=q                                    BxeԒH  �          @�Q��^�R?���  �K�\CL��^�R@dz��c33��CE                                    BxeԠ�  
�          @�
=?��H@~{@��HB?�\B���?��H?˅@��B�Q�BRQ�                                    Bxeԯ�  "          @�=q?�@�Q�@���B ��B�p�?�@"�\@��RB��B��f                                    BxeԾ:  
�          @�Q�?B�\@��@G�A�p�B��
?B�\@_\)@�(�B\�B�aH                                    Bxe���  T          @ə�?5@�p�@$z�A���B��?5@�Q�@���BF  B��
                                    Bxe�ۆ  	          @ə�?�{@���@	��A���B�33?�{@���@�B4=qB��H                                    Bxe��,  
�          @�녿��H@�z�@>{A�=qB�W
���H@e@�Q�BS��B�\)                                    Bxe���  T          @�=q?333@�Q�@
�HA�z�B�8R?333@�Q�@�{B7(�B�u�                                    Bxe�x  �          @ə�>�@�\)@"�\A���B�z�>�@��\@�Q�BEz�B�u�                                    Bxe�  "          @˅�=p�@���@�RA���B�z�=p�@�G�@�  B8
=Bǀ                                     Bxe�$�  �          @�  ���@�  �����RB�ff���@��@)��A�\)B�L�                                    Bxe�3j  �          @�\)���@��R�Q���ffB̅���@ə�?�R@��B�8R                                    Bxe�B  �          @�p����R@������\�\)B�{���R@�=q��33�n�RB��                                    Bxe�P�  T          @��H��p�@����8���ݮB�LͿ�p�@��þ�33�P��BϮ                                    Bxe�_\  �          @�=q��\@�(�������=qB����\@�p�?!G�@���B�{                                    Bxe�n  �          @ʏ\��z�@��������\BԮ��z�@�G�>�(�@{�BҨ�                                    Bxe�|�  T          @��2�\@��?L��@�{B�u��2�\@�G�@HQ�A�p�B�u�                                    BxeՋN  T          @�p��R�\@�?&ff@��B�z��R�\@�z�@8��Aי�B��R                                    Bxeՙ�  
�          @�(��n{@��;�p��Tz�B����n{@�=q?���A�33B�8R                                    Bxeը�  
�          @��H�g
=@�������RB���g
=@��
?�G�A�B�\                                    Bxeշ@  �          @˅�o\)@�  �����#�B����o\)@�Q�?��AffB��=                                    Bxe���  �          @���a�@�ff�Ǯ�f�HB����a�@�z�?+�@�(�B��H                                    Bxe�Ԍ  �          @����e@�G��J=q��z�C�)�e@����z��+33B�aH                                    Bxe��2  
�          @ə��}p�@`���U�� �
C
u��}p�@��H��ff�eG�C��                                    Bxe���  T          @���j�H@`���dz��\)CL��j�H@�{��G���  C �                                    Bxe� ~  �          @�  ���@#33�N{��CQ����@j=q��������C�\                                    Bxe�$  "          @�Q����\@!��U�(�C\���\@l�Ϳ�(���{C{                                    Bxe��  "          @�Q���Q�@@���U��33C���Q�@�(���\���\C�                                    Bxe�,p  �          @�Q����@ ���_\)��\Cٚ���@n�R�����CE                                    Bxe�;  �          @�����G�@#�
�Z=q���C�{��G�@p  ����=qCs3                                    Bxe�I�  
�          @�33��{@.�R�QG����HC�=��{@u������C��                                    Bxe�Xb  �          @��H��(�@(���A���
=C�{��(�@j=q��z��r�HC�                                    Bxe�g  
�          @�����p�@Tz��9�����C���p�@�\)��  �9G�CL�                                    Bxe�u�  �          @��H��Q�@Q��)����
=C��Q�@��\����'33C�                                     BxeքT  �          @�33�c�
@:=q�����$�C�3�c�
@�������p�C�                                     Bxe֒�  �          @��H�_\)@q��33���HC���_\)@�z����p�C xR                                    Bxe֡�  T          @Å�aG�@�>�33@VffB��aG�@�33@
=A�
=C                                      BxeְF  �          @\�r�\@���?�Q�A6{C��r�\@h��@<��A��C0�                                    Bxe־�  �          @�(��
=@�  �G����B��
=@��\?�ffAn�HB���                                    Bxe�͒  �          @ƸR�   @��{��B�\)�   @��>W
=?��HBٞ�                                    Bxe��8  �          @��
�^{@y����z���G�C���^{@�\)=�Q�?fffCL�                                    Bxe���  "          @�(�����@%������C
=����@$z�?�@��\C)                                    Bxe���  "          @�z����R@   ��G��>{Ch����R@333�����HC�H                                    Bxe�*  T          @��H��p�@;���G����RC5���p�@X�þ���z�HCc�                                    Bxe��  �          @�(����@h�������(�C�
���@����333��33C{                                    Bxe�%v  "          @�z����@Q��   ���HC}q���@L(������HQ�C^�                                    Bxe�4  �          @�����(�?�Q��1G���z�C@ ��(�@9���޸R���C}q                                    Bxe�B�  
�          @��
����?��\�N{� �C(������@\)�!����HC\                                    Bxe�Qh  T          @Å���
?��׿�����HC&&f���
@녿��0��C �                                    Bxe�`  �          @��H��  ?У׿�  ���C#xR��  @�Ϳp����
C33                                    Bxe�n�  �          @�=q��
=@l(�����33C
�{��
=@����������Cz�                                    Bxe�}Z  �          @\��=q@AG����H��  C����=q@c33�����RC�
                                    Bxe׌   "          @��H��
=@.�R����U�Ch���
=@C�
�8Q��
=C��                                    Bxeך�  
�          @�33���@I����(��8��C{���@XQ�>��?���CJ=                                    BxeשL  
�          @�������@l(��O\)��=qC�
����@l��?:�H@ᙚCxR                                    Bxe׷�  �          @�G�����@g��=p���  C������@g
=?G�@���C��                                    Bxe�Ƙ  �          @�G���ff@B�\�����FffC����ff@:=q?k�A�
C                                    Bxe��>  T          @�����G�@N{�.{��ffCh���G�@N{?+�@��
Cc�                                    Bxe���  "          @�=q��{@xQ쿴z��W33C
�f��{@��
>�  @��C	0�                                    Bxe��  �          @�G�����@Z=q��p��l(�C\����@QG�?��\A%C+�                                    Bxe�0  
�          @�Q���33@XQ�5��G�C����33@XQ�?333@�
=C��                                    Bxe��  	�          @������H@z=q��  �B{C
  ���H@��\>���@uC�q                                    Bxe�|  
�          @�
=��G�@*=q�u�
=C#���G�@{?�  A"{C�f                                    Bxe�-"  <          @������@P  ?��@��RC�R����@0  ?�A��C�                                    Bxe�;�            @�{���H?У�>�\)@.{C#� ���H?��?fffAC&\                                    Bxe�Jn  
�          @�Q����\@?\)����s
=CJ=���\@Vff�u��C@                                     Bxe�Y  �          @�Q���{@p  ��
���HC
=q��{@�  ��
=�\)C��                                    Bxe�g�  "          @������@Vff?\(�AG�C������@0  @
=A��C                                    Bxe�v`  �          @�����\@z�H�\)���RC	�\���\@u�?��
A!G�C
�                                    Bxe؅  "          @�����@o\)���R�iC����@���=���?uC�3                                    Bxeؓ�  �          @�������@Y�������C\����@q녾W
=�G�C
                                    BxeآR  "          @�\)���\@O\)�{�ĸRC+����\@|�Ϳ�G����C	�)                                    Bxeذ�  "          @�  ��Q�@O\)�*=q�ӅC����Q�@��׿�Q��733C��                                    Bxeؿ�  �          @�G���Q�@QG��Fff��G�C��Q�@�
=��=q�r�HC��                                    Bxe��D  "          @���y��@�ff�����O�
C���y��@�(�>��@�G�C��                                    Bxe���  �          @�(����@3�
������C33���@\�Ϳp���G�C�3                                    Bxe��  "          @����{@fff�Y����C�q��{@i��?(�@���C�)                                    Bxe��6  �          @�\)�C�
@�ff���R�L(�B�{�C�
@��R?��
A|��B��f                                    Bxe��  �          @�G��i��@��>.{?�C���i��@}p�?�A��\C��                                    Bxe��  �          @�ff�dz�@����\)�0��C��dz�@�33?��A�  C�                                    Bxe�&(  �          @�p��4z�@���?B�\@���B�G��4z�@��@!�A�\)B���                                    Bxe�4�  T          @������@�33@  A˅B��ῇ�@X��@uB={B���                                    Bxe�Ct  T          @�녿��
@��@��A�z�B��
���
@\(�@w�B<��B�=q                                    Bxe�R  �          @�\)��ff@�
=?���A�ffB�8R��ff@x��@g
=B%��B��H                                    Bxe�`�  "          @�����p�@�z�?ǮA��B�#׿�p�@z�H@S�
B  B鞸                                    Bxe�of  �          @���=q@��?��AW�
B�8R�=q@|��@AG�B�B�R                                    Bxe�~  "          @����0��@���>Ǯ@y��B�=�0��@�  @�A�z�B�
=                                    Bxeٌ�  �          @��\�%�@��u�!G�B��
�%�@�=q?�z�A�(�B�=q                                    BxeٛX  l          @�z��6ff@���k��33B�L��6ff@�=q?�(�A��RB�8R                                    Bxe٩�  �          @���J�H@c33�AG��ffC���J�H@������m�B�(�                                    Bxeٸ�  �          @��'�?��R����`��C�
�'�@l(��s33��HB��                                    Bxe��J  �          @�\)�XQ�@�����H�933C�H�XQ�@w
=�G
=��  C8R                                    Bxe���  "          @����Fff@>{��z��/�
C0��Fff@�=q�.{����B�ff                                    Bxe��  �          @���B�\@8Q���  �5�
C���B�\@����7
=���B��                                    Bxe��<  
�          @���A�@
=����H�C&f�A�@z=q�X���	z�B��\                                    Bxe��  �          @�\)�L(�@QG��tz��p�CB��L(�@�
=��
����B��                                    Bxe��  "          @����P��@1��s33�&Q�C�\�P��@\)��R���C0�                                    Bxe�.  �          @��R�Z=q@"�\�qG��&=qCJ=�Z=q@p���#33��G�C(�                                    Bxe�-�  �          @����S33@(���  �0�C��S33@p���333��{CT{                                    Bxe�<z  T          @��H�"�\@(Q�����Kp�C  �"�\@���L(����B�k�                                    Bxe�K   �          @�=q�>{@1G����
�5��C���>{@�33�333��\)B�                                    Bxe�Y�  �          @����xQ�@��g����CǮ�xQ�@QG��%��=qC�
                                    Bxe�hl  �          @�G��@��@��\)�?�
C
�@��@o\)�Dz��33C �\                                    Bxe�w  �          @����@��@���p��<ffC)�@��@s33�>�R��ffC p�                                    Bxeڅ�  T          @����(�@R�\��=q�3G�B�\�(�@����$z��ә�B�G�                                    Bxeڔ^  �          @����U�@'��r�\�'Q�C��U�@u��#�
��Q�C��                                    Bxeڣ  �          @����@��@<���w
=�*{C���@��@��� ����ffB���                                    Bxeڱ�  "          @����*�H@Dz�?Q�A5��C�q�*�H@#�
?��A�{C33                                    Bxe��P  "          @�p��(��@|��@%A�33B�z��(��@.�R@w
=B5�C�                                    Bxe���  �          @��H�B�\@�Q�?���A���B�k��B�\@L��@3�
B�C�=                                    Bxe�ݜ  �          @�\)�p�@�=q?�  A��\B왚�p�@g�@R�\B��B�ff                                    Bxe��B  "          @�\)�p�@���?\(�A�B�\�p�@�z�@!�A�{B�                                    Bxe���  �          @�Q��33@�������B���33@��?�\)Ak�B�{                                    Bxe�	�  �          @�
=���H@�=q�fff�  B�.���H@���?�  A*{B�Q�                                    Bxe�4  
Z          @�
=��(�@��Ϳ(������B��׾�(�@�G�?�  A]�B��
                                    Bxe�&�  
�          @���?0��@��
?+�@�G�B�L�?0��@���@#�
A��B�u�                                    Bxe�5�  T          @�{<#�
@���?�Q�AB�\B�<#�
@�  @G
=B  B��R                                    Bxe�D&  �          @�z�?L��@���?���A^{B�\?L��@��@L��B
�B���                                    Bxe�R�  �          @�(�?z�@�@!�A�B��3?z�@l��@�(�B?33B�W
                                    Bxe�ar  "          @��H�aG�@���@XQ�B�RB̮�aG�@"�\@�33Bn��B�(�                                    Bxe�p  "          @����Z�H?xQ�@��\BC�
C$8R�Z�H���@���BH��C=5�                                    Bxe�~�  �          @���O\)@^�R@-p�A�C�3�O\)@�@q�B.p�C��                                    Bxeۍd  �          @�ff�,(�@���?G�A�RB�ff�,(�@|��@A�z�B�k�                                    Bxeۜ
  "          @��R�{�?�Q�@HQ�B  C���{�?5@g
=B(��C)�\                                    Bxe۪�  �          @�G��w�@G�@S�
Bp�Cff�w�?8Q�@s�
B0(�C)p�                                    Bxe۹V  �          @�G��h��@1�@G
=Bp�C���h��?�G�@x��B2p�C�=                                    Bxe���  T          @�\)�R�\@8Q�@G�B��C
�R�\?���@{�B<G�C{                                    Bxe�֢  
�          @����mp�?��@N�RB(�C�mp�?!G�@l(�B1��C*Y�                                    Bxe��H  
�          @���!G�@Mp�@p  B*z�C (��!G�?�
=@�z�Bcp�CB�                                    Bxe���  �          @�=q�33@fff@j�HB&G�B�z��33@�@�ffBh�C��                                    Bxe��  T          @��Ϳ�@s�
@h��B!�B���@�\@��Bg
=C�3                                    Bxe�:  �          @��R��G�@���@^{B��B�aH��G�@3�
@�Q�Bgp�B�aH                                    Bxe��  
�          @�=q�   @���@VffB=qB�=q�   @L��@�Q�B_33BŽq                                    Bxe�.�  "          @�33��?�(�@Mp�BVG�C����<�@\(�Bm�RC3&f                                    Bxe�=,  �          @��x�ÿ�(�@_\)B�
CN��x���Dz�@%A�
=CZB�                                    Bxe�K�  T          @�
=���\�G�@Y��B�CNff���\�E�@\)A��CY
=                                    Bxe�Zx  
�          @�Q��xQ쿚�H@�Q�B2p�CET{�xQ��!G�@VffB\)CU�                                    Bxe�i  
�          @�p��s�
�#�
@�(�B=(�C4�{�s�
���H@w
=B-�CH��                                    Bxe�w�  
�          @����u�?�\)@��HB4�\C aH�u���@��BA\)C5޸                                    Bxe܆j  T          @��\���H?���@x��B(=qC!L����H�#�
@�(�B533C4��                                    Bxeܕ  "          @��H��\)?�z�@n{BffC!����\)=L��@~�RB-  C3W
                                    Bxeܣ�  
�          @���n�R@(�@e�B(�Cٚ�n�R?��@�ffB={C${                                    Bxeܲ\  �          @�=q�Q�@y��@1�A���C��Q�@-p�@|(�B+G�Cc�                                    Bxe��  T          @��@  @�33@��A�B�k��@  @AG�@l��B#�C�{                                    Bxe�Ϩ  
Z          @���p�@�ff?�Q�A��HB�Q��p�@p��@Z�HBG�B�z�                                    Bxe��N  
�          @����33@�ff@G�A���B����33@xQ�@s�
B �HB�8R                                    Bxe���  
(          @���ff@���@�HA���B�8R�ff@{�@\)B'=qB�L�                                    Bxe���  
�          @�p����H@l��@��HB9z�B�Q쿚�H@ff@�33B�{B�                                    Bxe�
@  T          @����@j�H@��HB9�B�Ǯ���@�@��\B���B�Ǯ                                    Bxe��  T          @����G�@z�H@\(�Bz�B�
=��G�@!�@��Bd��B�                                    Bxe�'�  
Z          @����   @�@*=qA��HB�B��   @aG�@��\B5
=B�G�                                    Bxe�62  �          @���
=q@�\)@�AȸRB�\�
=q@h��@xQ�B*
=B�W
                                    Bxe�D�            @��
���@��R@�A��B�p����@�  @dz�B�HB�                                    Bxe�S~  �          @��H��\@��?�\@���B�k���\@��
@ffA��B�33                                    Bxe�b$  �          @��H�#33@�G�>aG�@B�u��#33@���@33A�\)B�R                                    Bxe�p�  T          @�{�\)@�33>�G�@�G�B��f�\)@���@(�A�=qB��)                                    Bxe�p  T          @Å�X��@�=q>B�\?�\B�z��X��@�
=?�{A�  B�L�                                    Bxeݎ  "          @����@��?xQ�A�B�8R��@�33@3�
A��B�.                                    Bxeݜ�  T          @����;�@�?�z�AT��B����;�@�@B�\A�z�B�                                    Bxeݫb  
�          @�33�E@�z�?�33A-G�B��E@�\)@1�A�ffB�B�                                    Bxeݺ  �          @�����@��
�k��1G�B�=q���@~{?��Aa�B��                                    Bxe�Ȯ  �          @�G��tz�?�(��\(���
C���tz�@?\)�'
=��{C�                                    Bxe��T  �          @�Q��@��@#33�o\)�-C���@��@h���,�����
C�)                                    Bxe���  �          @�ff�fff>L����{�EQ�C0�
�fff?Ǯ�y���3\)C�{                                    Bxe���  �          @�{�R�\?&ff�����U��C(��R�\@��=q�8C�=                                    Bxe�F  
�          @�p���(�?8Q��q��(  C*\��(�?�z��U��HC&f                                    Bxe��  
�          @�����?���a��z�C#�����@��:�H�C.                                    Bxe� �  "          @�  ��ff?n{�`�����C(8R��ff@G��AG���Q�C�)                                    Bxe�/8  "          @��\����6ff��=q�<��CU}q����ff��Q����RCP��                                    Bxe�=�  
�          @�Q��}p����R�C�
���CEc��}p��B�\�R�\���C6�3                                    Bxe�L�  �          @�(��|��?����5��C���|��@��{���C��                                    Bxe�[*  �          @�=q�,(�@���\(��.�
C
W
�,(�@W��\)��{C �{                                    Bxe�i�  �          @�G��  @#33�u��Ap�Cc��  @i���3�
�  B�aH                                    Bxe�xv  T          @���}p�@*�H�0  ��C���}p�@Z=q��  ��z�C:�                                    Bxeއ  �          @�z����@C33��R��33C�\���@fff����<  C(�                                    Bxeޕ�  T          @��R��p�@#�
>��?ٙ�C���p�@Q�?uA,��C��                                    Bxeޤh  T          @����(�@,��?uA0(�C�
��(�@��?�\A���CaH                                    Bxe޳  �          @�ff?8Q�@o\)�s33�`z�B�W
?8Q�@vff>�\)@�(�B��                                    Bxe���  T          @�����
@��\��Q�����B�.���
@���
=q��p�B���                                    Bxe��Z  
�          @����ff@i���,(��  B�LͿ�ff@�녿�33��{Bգ�                                    Bxe��   
�          @�p���\@%�fff�>{C 33��\@e�&ff� ��B�(�                                    Bxe���  �          @�{��
=@333�c33�8�B���
=@q��\)��p�B�                                    Bxe��L  �          @�=q�%@HQ��Tz����C���%@�Q��
�H�Ù�B���                                    Bxe�
�  T          @��R�Fff@.�R�c�
�#33C
���Fff@mp��!���G�C�                                    Bxe��  
�          @�Q��@Z�H�X���$z�B� ��@���	���ŅB�8R                                    Bxe�(>  �          @�=q�E�@(Q��4z��ffC� �E�@W���{��G�C\)                                    Bxe�6�  �          @��׿p��@��\�#�
�z�B̔{�p��@�(�?�ffA��RBͽq                                    Bxe�E�  �          @�z�\)@��@J=qB�Býq�\)@8Q�@���B]��B�\                                    Bxe�T0  �          @�{���@��R>�?���B��þ��@��?޸RA�33B�z�                                    Bxe�b�  �          @�(���?�=q@��B��B�� ����\)@��B�(�CtW
                                    Bxe�q|  �          @�Q�?�  ?#�
@�G�B���A�=q?�  ����@�
=B�(�C�G�                                    Bxe߀"  �          @���.{?���@��B�8RB�=q�.{���R@�p�B���Cq+�                                    Bxeߎ�  �          @��R�u?�\)@�(�B��B��u����@��
B��CK�                                    Bxeߝn  
�          @�  ��ff@�=q@\(�B��B�z��ff@S33@�p�BP  B�W
                                    Bxe߬  T          @����{@�G�?�(�A�
=B�Ǯ��{@�{@_\)B�HB�                                    Bxeߺ�  T          @�z��   @���?��RAlQ�B�p��   @�=q@AG�A��B�(�                                    Bxe��`  �          @�(���Q�@�z�>��H@��HB����Q�@�  @�A�=qB�.                                    Bxe��  T          @�\)�z�@�p��:�H�陚B�\)�z�@���?^�RA��B�=                                    Bxe��  "          @������@�p��u��B�
=���@�
=?&ff@��B�                                    Bxe��R  �          @�����@�ff���
�Q�B�W
��@��>�=q@0  B��                                    Bxe��  �          @�G��@�G������
B���@�{�
=q���B���                                    Bxe��  �          @����Q�?�������B��)��Q�@:�H��z��fz�B���                                    Bxe�!D  �          @�����<���ff�C2�����?�z���{��C G�                                    Bxe�/�  "          @�������?�  ����3C
uÿ���@'
=��Q��q�B��                                    Bxe�>�  �          @��H���#�
���\¦{C�=q��?z�H����B�{                                    Bxe�M6  �          @���!G�@?\)�g��+33C��!G�@{��#�
��Q�B�G�                                    Bxe�[�  �          @�33�fff@vff����C\�fff@�(������.�HCs3                                    Bxe�j�  �          @��
����@hQ쿔z��:ffC������@s�
����\)C
:�                                    Bxe�y(  �          @�33���\@e������
C�3���\@e�?\)@���C�q                                    Bxe���  �          @�����z�@Tz῰���YCn��z�@e���{�UC\)                                    Bxe��t  �          @�{���@333�
=��C�q���@W���
=�_�
C)                                    Bxe�  �          @�  ���@#�
�2�\��p�C8R���@P�׿�z�����C��                                    Bxe��  �          @�p���(�@%�H�����HC��(�@Y���\)����C��                                    Bxe��f  T          @����w�@ff�}p��$�C��w�@Y���Fff��Q�C
�3                                    Bxe��  T          @�\)�xQ�@z�����4G�C���xQ�@QG��g
=��C��                                    Bxe�߲  �          @�����Q�?�ff�1G�����C(�q��Q�?��ff��33C!                                    Bxe��X  T          @������H?����\��33C&�=���H?������IG�C"33                                    Bxe���  T          @����H?�\)�Z�H��RC&G����H@�
�<����33C��                                    Bxe��  
�          @�������@C�
�33����Ck�����@e�����T(�C�R                                    Bxe�J  k          @����q�@l(�������C���q�@�{�����0��C                                    Bxe�(�  "          @�=q�0��@���0����p�B���0��@�
=?E�@��RB�
=                                    Bxe�7�  T          @��\��@���?�\@��B垸��@�?�
=A�=qB��f                                    Bxe�F<  �          @�>�\)@HQ�@��Bg{B�Ǯ>�\)?���@�p�B��=B���                                    Bxe�T�  �          @�ff����@u@���BF�\B��;���@��@�{B�L�B��                                    Bxe�c�  �          @�
=�u@��R@��RB3B��R�u@5@�\)Bu�B��\                                    Bxe�r.  �          @��
��@|(�@�=qB>�RB����@#33@�Q�B�\)B��                                    Bxe��  �          @���k�@E�@��HBk33B����k�?�G�@�Q�B�W
B�u�                                    Bxe�z  T          @��þu@tz�@���BG33B�8R�u@Q�@�p�B�W
B��\                                    Bxe�   �          @�33?Tz�@���@j�HB�B�aH?Tz�@c33@��
BT��B���                                    Bxe��  "          @�33>�=q@�Q�@<��A��
B�{>�=q@�
=@��B633B���                                    Bxe�l  �          @�녽�\)@�=q?�{Av�HB�ff��\)@�(�@J�HA��B���                                    Bxe��  "          @�ff�&ff@��?�33A�
=B�B��&ff@�(�@H��B�B��                                    Bxe�ظ  "          @��R��\)@���?��HA@z�B��)��\)@��@-p�A�G�B�W
                                    Bxe��^  "          @�ff��Q�@�녿˅�|(�B�\��Q�@�G��L�Ϳ   Bڀ                                     Bxe��  "          @��Ϳ333@��R@�A�33B�#׿333@��H@n{B'z�Bǅ                                    Bxe��  T          @�z��.{@Tz��(Q����CT{�.{@z�H�������RB��                                    Bxe�P  T          @����5@o\)�����=qB�W
�5@�{���\�6�HB�8R                                    Bxe�!�  "          @����+�@�ff?O\)A�
B��H�+�@���@	��A�B�8R                                    Bxe�0�  �          @�G���R@�z�@'
=A��HB�����R@`  @n{B#��B��                                    Bxe�?B  
�          @�33�aG�@H��@Z�HBC
T{�aG�@�@���B5�RC��                                    Bxe�M�  �          @��H��
@{@p  BE�C���
?�{@��RBj�
Cc�                                    Bxe�\�  �          @�  �Ǯ?W
=@�=qB�{B�Q�Ǯ�8Q�@��\B�W
Cq(�                                    Bxe�k4  
(          @��R��
=?�\@���B�C%E��
=�u@��B�G�CNu�                                    Bxe�y�  �          @�(��ٙ�>��R@��\B�  C)�\�ٙ���z�@��RB��)CV.                                    Bxe∀  T          @�(�����Y��@���B���CU\�����\@��RBw��Cp+�                                    Bxe�&  "          @��\��H�#�
@���B{�
CB�\��H��Q�@�G�B^�CZ��                                    Bxe��  �          @��H�z=q@ ��@,(�A���C^��z=q?�
=@QG�B{C��                                    Bxe�r  �          @��R>�@/\)@VffBIffB��)>�?�p�@~{B�W
B�(�                                    Bxe��  �          @��\>��H@HQ����R�S�B�.>��H@����Mp����B��                                     Bxe�Ѿ  �          @�=q>Ǯ@{��\)�}33B�G�>Ǯ@U�n{�?z�B���                                    Bxe��d  "          @�����G�?\(�@���B�ǮB��ý�G���R@�B�C�ٚ                                    Bxe��
  �          @���"�\?�ff@�Q�Bo��Cs3�"�\���@��
ByQ�C9Ǯ                                    Bxe���  
�          @����g�?���@y��B7�\C#&f�g�<��
@���B@�HC3�                                     Bxe�V  �          @�ff�Y��?�=q@n{B/��C�R�Y��?E�@��\BE�C'8R                                    Bxe��  "          @��\�p�?�
=@�(�BZ33C
���p�?E�@��By�
C �f                                    Bxe�)�  "          @��\����?��@�ffB|��B�𤿹��?�@���B���CT{                                    Bxe�8H  "          @�33��Q�?�{@��B�{C�ÿ�Q�>��@��B���C)�H                                    Bxe�F�  �          @�녿s33?޸R@��B�
=B�G��s33>\@�ffB��3C�                                    Bxe�U�  "          @�ff��\)?�Q�@��RB�L�C�H��\)>��@�p�B��qC-��                                    Bxe�d:  
�          @��H���@Q�?n{A1�C����@33?\A��\CxR                                    Bxe�r�  
Z          @�z����\@A�>���@[�C\)���\@6ff?��AEp�C{                                    Bxeう  "          @�{�~�R@I���^�R��HC�R�~�R@P�׼#�
���C�3                                    Bxe�,  T          @�p�����@2�\��=q�C33C
����@>�R���
�c33C\)                                    Bxe��  �          @����z�H@(��G����HC
=�z�H@8Q쿧��q��C�H                                    Bxe�x  �          @�z��aG�?��H���G�Cc��aG�@*�H�!�����C�{                                    Bxe�  �          @�{�i��@7���Q���\)C�q�i��@Q녿����O�C
�                                    Bxe���  �          @�ff�`��@n{    �#�
CL��`��@fff?uA-�CJ=                                    Bxe��j  �          @�ff�Z=q@s�
�L�����C�)�Z=q@n�R?J=qAffCs3                                    Bxe��  �          @��H�W�@���
=����C\�W�@'
=��ff��CJ=                                    Bxe���  
�          @��H�Fff?���q��D��C!u��Fff?�(��X���*�RC�                                     Bxe�\  �          @�Q��tz�@33�1����C��tz�@;��ff��Q�Cs3                                    Bxe�  �          @�
=�dz�@   �H����
C�R�dz�@/\)�!G���(�C��                                    Bxe�"�  
(          @�=q�޸R@
=q��ff�bG�C ��޸R@J=q�`���0��B���                                    Bxe�1N  "          @�  ���
@�������o(�B��
���
@QG��c33�8
=B��)                                    Bxe�?�  �          @�����z�?:�H��R��G�C*� ��z�?��
��(�����C#Ǯ                                    Bxe�N�  �          @�=q���?k���33��
=C(+����?��׿�{��\)C"�                                    Bxe�]@  T          @�
=���
��>�{@�Q�CG�����
���H��Q쿅�CHE                                    Bxe�k�  �          @�{��ff��\)���
�uCC}q��ff���þ�p����HCB��                                    Bxe�z�  T          @���\)�   �\)��G�CQ!H��\)�Q�B�\�  CP                                      Bxe�2  �          @����
=@   �!G���\CY���
=@%�<��
>B�\C��                                    Bxe��  "          @���qG�@tz῅��0  C���qG�@}p��L�Ϳ�C�
                                    Bxe�~  
�          @�Q���@9�����H�N{C� ��@G
=��ff���C��                                    Bxe�$  �          @����(�@��У���z�C���(�@�H����0  C��                                    Bxe���  
�          @�Q���ff@*�H������C  ��ff@@  �u�!C��                                    Bxe��p  �          @�\)���?�z��33���\C!
���@���
=�r{C�=                                    Bxe��  T          @������@   �����C�q����@(���p��{\)CQ�                                    Bxe��  T          @�Q���33?�zῃ�
�0  C$����33?�\)�&ff��p�C"ff                                    Bxe��b  =          @������R>�(��#�
��p�C/�����R?녿����
C.                                    Bxe�  k          @������=�\)��
=���C3J=����>.{�Ǯ��Q�C2@                                     Bxe��  T          @�  ���>�  �\(���
C1c����>�ff�E��G�C/B�                                    Bxe�*T  
�          @�\)��=�Q������ffC3  ��>B�\��Q��tz�C2�                                    Bxe�8�  
(          @�ff��{>��L�Ϳ   C2�)��{>\)�#�
�\)C2��                                    Bxe�G�  
�          @��R��ff=�Q쾏\)�>�RC3���ff>#�
��  �)��C2T{                                    Bxe�VF  
(          @��R��p�    ������C3�q��p�=�����(�C2�                                    Bxe�d�  "          @�Q���\)=�\)�   ��C38R��\)>B�\�����HC1��                                    Bxe�s�  	�          @�����>��ÿ���z�C0W
���?B�\���H����C+��                                    Bxe�8  
�          @�ff���<��
=q���C3�����?\)���
=C-��                                    Bxe��  "          @�ff��z��\��z���ffC9�R��z���Ϳ�p���
=C5�                                    Bxe埄  T          @�{��(�<��
�У����C3� ��(�>�녿�=q��Q�C/c�                                    Bxe�*  �          @�p���{�L�Ϳ��\�X��C<�q��{��׿��s�
C90�                                    Bxe��  
Z          @�p����>�{��(��QC0J=���?�R��{�=p�C-:�                                    Bxe��v  �          @�������?�ff�z�H�(z�C(n����?�G��333��C&J=                                    Bxe��  T          @�����G�?#�
�L�����C-!H��G�?O\)�!G���C+J=                                    Bxe���  	.          @�(���{��������E�C6�H��{<��
��z��J=qC3�{                                    Bxe��h  �          @��H��33��ff��z��H(�C?����33�@  ��\)�n�\C<T{                                    Bxe�  �          @��\��=q��  ��(��S�
C?.��=q�0�׿��x  C;��                                    Bxe��  
�          @��
���ÿ��@  �   C9����þ�{�Y���=qC7��                                    Bxe�#Z  �          @����=q�
=q=u?!G�C9����=q�
=q��\)�@  C9Ǯ                                    Bxe�2   
�          @�z���33�   <�>�\)C9O\��33���H��Q�z�HC9:�                                    Bxe�@�  
(          @���33�Y���#�
��C=\��33�Q녾u�$z�C<�3                                    Bxe�OL  �          @��
��  �E��=p����RC<aH��  �z�fff�
=C:G�                                    Bxe�]�  
�          @�(���
=���
�.{��  C?+���
=�W
=�fff�=qC=0�                                    Bxe�l�  "          @��������5�#�
��33C;�������
=q�J=q��HC9��                                    Bxe�{>  �          @�(���33���L�;�C8޸��33��(��������C8�H                                    Bxe��  �          @�=q��G���33=L��?
=qC7���G���33����33C7�=                                    Bxe昊  T          @�����׾Ǯ>�
=@���C85����׾��>��
@]p�C9&f                                    Bxe�0  �          @��H������\>�z�@FffC9�=������>#�
?޸RC:�                                    Bxe��  �          @�z����
���ͼ����
C5\���
��Q�L�Ϳ��C4��                                    Bxe��|  �          @������>�Q�>�{@i��C0�����>�=q>��@�ffC1                                    Bxe��"  T          @�p���=q?L��?�  A��\C+  ��=q>\?��A��HC/��                                    Bxe���  
�          @�(�����?fff?�{A���C)�����?   ?��
A�=qC.J=                                    Bxe��n  
�          @��
��  >��
?��
A1G�C0����  =�\)?�=qA9G�C35�                                    Bxe��  �          @����33>��?
=@���C/�{��33>�=q?+�@��
C1!H                                    Bxe��  �          @�p���  ?�G�>#�
?�C&���  ?�Q�>�G�@�p�C':�                                    Bxe�`  T          @�����H?@  >�33@l(�C+�����H?&ff?�\@��C-                                    Bxe�+  
�          @�p����?J=q�#�
��C+�H���?L��<��
>��C+xR                                    Bxe�9�  �          @����{?+�>aG�@��C,����{?(�>�Q�@r�\C-��                                    Bxe�HR  
�          @�����R?
=q>�?��C.L����R?   >�  @#�
C.�q                                    Bxe�V�  T          @����ff>�>��@���C/&f��ff>�Q�?�\@���C0=q                                    Bxe�e�  �          @�G����R>Ǯ��
=��(�C/�����R>����
�c33C.                                    Bxe�tD  �          @�\)���H<#�
�   ����C3�{���H?��(���(�C,޸                                    Bxe��  �          @�
=���R����>{��HC<&f���R=��
�A���C2�                                    Bxe瑐  �          @����
=���7
=���C:����
==��9����C2h�                                    Bxe�6  
�          @�������������p�C6�q���>u�����C1@                                     Bxe��  �          @�����\��G��=q��Q�C5T{���\>�(��Q�����C.�                                    Bxe罂  "          @����\�����5����C8#����\>��R�5���C0+�                                    Bxe��(  �          @�z�����>�����������C0������?0�׿��H���\C,(�                                    Bxe���  �          @�(�����L���(��ٮC6c����>�33��H��{C/�\                                    Bxe��t  "          @����
=?��$z���C-����
=?��������z�C'0�                                    Bxe��  T          @����  ?&ff��z���(�C,�{��  ?����(�����C(�                                    Bxe��  T          @�z���(�>�
=�
=q��ffC/\��(�?c�
� �����HC)�3                                    Bxe�f  �          @�p����?(���
��Q�C,�����?��������RC'B�                                    Bxe�$  �          @��
���
?}p���  �.=qC)
=���
?�Q�B�\�  C&�3                                    Bxe�2�  �          @�
=���\?L�ͿG��z�C+� ���\?s33�
=��\)C)�f                                    Bxe�AX  �          @�����Q�?��p���Q�C-����Q�?�  ��\��Q�C(!H                                    Bxe�O�  T          @�p��}p�?W
=�<���G�C'���}p�?�(��,���(�C�3                                    Bxe�^�  �          @����{?8Q��G����HC+L���{?�Q���
����C%��                                    Bxe�mJ  �          @�
=��(�>�\)�������C0����(�?+���(����
C,5�                                    Bxe�{�  �          @��R��Q�#�
�����p��C4���Q�>�\)�����k�C0�=                                    Bxe芖  �          @�{������
��G�����C7�����=�\)�����G�C3(�                                    Bxe�<  �          @�ff����@  ��=q���HC<�q����Ǯ���H��z�C8�\                                    Bxe��  T          @�����ff��  �h���#
=C?p���ff�G������FffC<�q                                    Bxe趈  =          @�����������H�\��C7z�������  �c\)C4^�                                    Bxe��.  k          @�G���ff�aG��:�H�ffC6�{��ff�u�B�\�(�C4��                                    Bxe���  �          @�(����ü��s33�*�\C4Y�����>.{�n{�'�
C2
=                                    Bxe��z  �          @������#�
�aG���C;!H�����G���  �0(�C8�H                                    Bxe��   �          @�����33�0�׿�  �/33C;�q��33��׿�\)�Ep�C95�                                    Bxe���  �          @�����(��   ����Q�C:+���(�<#�
�Q��ٙ�C3�                                    Bxe�l  �          @�  ���\��(���33�x��CJ+����\���ͿE���RCH�                                    Bxe�  T          @�����녿��\�!G���z�CB)��녿�{�aG��33C@\)                                    Bxe�+�  "          @����ff���
��
=����CE&f��ff��33�=p��{CC�
                                    Bxe�:^  "          @��H��ff��Q�=�G�?��RCJu���ff�������Z�HCJ=q                                    Bxe�I  �          @�z����>�p�����(�C/�=���>�׾�(����C.��                                    Bxe�W�  
�          @��������#�
���C5c�������#�
��G�C5Q�                                    Bxe�fP  �          @�����{�����(����CC�q��{�fff�p���Q�C>�)                                    Bxe�t�  
�          @����ff������(���ffCA����ff�^�R��Q���=qC=�                                    Bxe郜  T          @�33�����R���
�^ffCD�{����Q�Ǯ����CA�
                                    Bxe�B  �          @���������G��z�H�0  CE���������
���\�f{CB��                                    Bxe��  �          @�����G��L�Ϳ���=C6E��G�<���=q�A�C3�3                                    Bxe鯎  
�          @������H�#�
���O
=C4&f���H>k���33�J�HC1ff                                    Bxe�4  T          @����녿h�ÿ�(��\)C>33��녿z�У���C:�
                                    Bxe���  T          @��H���
��z�s33�#\)C@���
�p�׿��J{C>aH                                    Bxe�ۀ  �          @�G���(��Tz�z�H�+�
C=(���(���������FffC:�R                                    Bxe��&  
�          @�G���\)�
=q�z���  C9�)��\)��녿+���\C8p�                                    Bxe���  �          @�G���  ��G����R�XQ�C8����  ��p���������C8                                      Bxe�r  �          @�����H>aG�>�
=@��C1}q���H>�>�ff@��\C2��                                    Bxe�  
�          @�����
>u?�p�A���C15����
��\)?�  A��\C4ٚ                                    Bxe�$�  T          @��\��
=�Tz�=u?8Q�C=xR��
=�Q녽�G����RC=k�                                    Bxe�3d  �          @������\���ÿL���ffC@�����\�c�
��  �8Q�C>k�                                    Bxe�B
  
�          @�{����>.{?���A@z�C2����ýL��?�=qAB�RC4�
                                    Bxe�P�  
�          @�(������Ǯ>�(�@�33C8h��������>�{@uC9O\                                    Bxe�_V  
�          @�\)��z�!G��.{�G�C;Q���z�zᾙ���^{C:�                                     Bxe�m�  T          @�Q����
�(�ÿJ=q�G�C;�����
���fff�'33C9�f                                    Bxe�|�  T          @����G���Ϳ����C:0���G���(���R��{C8�H                                    Bxe�H  �          @����=q����  �1G�C9+���=q���;����p��C8�                                     Bxe��  
Z          @��
��ff������G��dQ�C7s3��ff�#�
��ff�j�\C4k�                                    Bxeꨔ  �          @��R��
=���ÿ�G����HC7�)��
=���
��ff��Q�C4@                                     Bxe�:  �          @�z���
=�\)���
�g�C5����
==����
�h  C2�H                                    Bxe���  
�          @�33�����  �xQ��/�C6�{����L�Ϳ�  �5G�C4�                                    Bxe�Ԇ  T          @�������>L�ͿTz����C1�����>�p��E��
ffC/�
                                    Bxe��,  �          @����{?G���ff�=��C*�q��{?z�H�^�R���C(�=                                    Bxe���  T          @�z���  ?�G�����C(����  ?�{��Q���Q�C'��                                    Bxe� x  T          @�����  ?Y���^�R�(�C*h���  ?�  �.{��z�C(��                                    Bxe�  �          @�33��\)?5�8Q��C+���\)?Tz�\)�ʏ\C*}q                                    Bxe��  T          @��H���?녿5��C-�����?333�z��ӅC,�                                    Bxe�,j  T          @��H��\)?s33�����33C)8R��\)?�G��W
=�
=C(��                                    Bxe�;  �          @��H��?�����  �5�C&W
��?�(��#�
�uC&\                                    Bxe�I�  �          @����{@ �׽�\)�G�C�\��{?�p�>���@o\)C
                                    Bxe�X\  �          @�33��=q@(��>�{@x��C�\��=q@!G�?^�RA��C�                                     Bxe�g  �          @�  �hQ�@C�
?�{A�Cٚ�hQ�@.�R?�Q�A�G�C\                                    Bxe�u�  
�          @�p����\@1G���\)�I��Cn���\@1G�>�=q@B�\Ck�                                    Bxe�N  T          @����l(�@*=q?
=@��C:��l(�@\)?�{A_�C�                                    Bxe��  
�          @�  >W
=@���?���A�B���>W
=@��@3�
B
��B�                                      Bxe롚  �          @��\?��@�=q?�=qA��\B��?��@�
=@p�A��B��3                                    Bxe�@  �          @�\)���
@?z�A@��Cc׿��
?�
=?}p�A���C�)                                    Bxe��  "          @�33��  =u?5A ��C3G���  ���
?5A Q�C4�f                                    Bxe�͌  �          @�G����?�=q?��@��C'xR���?p��?L��A=qC)
=                                    Bxe��2  "          @�=q��?�=q>�p�@�C'����?}p�?z�@У�C(��                                    Bxe���  �          @�(���p�?�  ?#�
@�ffC%�R��p�?���?aG�A��C'c�                                    Bxe��~  �          @�z���p�?�Q�?O\)Az�C&n��p�?�G�?��\A8(�C(p�                                    Bxe�$  �          @�z�����?��\?��AL��C(33����?G�?���Am�C*�3                                    Bxe��  �          @��\����?�\)?s33A,��C$\����?�33?���AZ�HC&p�                                    Bxe�%p  �          @�G�����?��
?fffA%C$������?�=q?��AQG�C'B�                                    Bxe�4  �          @�����33@�?n{A*ffC����33@ff?�\)A|��C+�                                    Bxe�B�  �          @�G���Q�@$z�?L��AffC�f��Q�@
=?��Al��C��                                    Bxe�Qb  �          @��
��Q�@+�?(�@�RCG���Q�@ ��?�\)AUp�C                                      Bxe�`  T          @�33�~�R@)��?B�\A�
CL��~�R@��?�G�AqG�CT{                                    Bxe�n�  T          @��R�^�R@C33?�Q�A��C
���^�R@-p�@   A�G�C!H                                    Bxe�}T  �          @����G�@Tz�@�\A��HC(��G�@7
=@)��B �C	n                                    Bxe��  T          @����Mp�@X��?޸RA�33Cs3�Mp�@?\)@
=A�z�C	�                                    Bxe욠  �          @�
=�P��@U?�{A�ffC\)�P��@@��?��RA���C	T{                                    Bxe�F  �          @�G��Y��@9��@�\A�\)C���Y��@��@3�
B33CǮ                                    Bxe��  �          @�
=�X��@Fff?�z�A�ffC	�
�X��@*�H@{A�ffC�=                                    Bxe�ƒ  �          @�Q��S�
@@  @�A�(�C	���S�
@"�\@%�A���Ch�                                    Bxe��8  �          @�G��\��@fff>L��@�\C� �\��@_\)?c�
A$��C�)                                    Bxe���  �          @���z=q@8�ü#�
��C���z=q@5�?��@θRC�                                    Bxe��  T          @�G��X��@QG���
=���C
�X��@a녿��\�<��C�)                                    Bxe�*  T          @��H�u�@5���
=���
C}q�u�@Fff��{�JffC�3                                    Bxe��  �          @�z���  ?����ff�lz�C$G���  ?Ǯ���\�8z�C!�=                                    Bxe�v  �          @�Q��Q�@w
=@�A��HB�aH�Q�@X��@.�RB��B��                                    Bxe�-  �          @���!G�@|��?��A��
B�.�!G�@`��@'�A�\)B�k�                                    Bxe�;�  "          @�����@�  ?��RA�
=B�����@b�\@.{B  B��                                    Bxe�Jh  �          @�  �ff@��\?��A��RB�u��ff@h��@(Q�B G�B��                                    Bxe�Y  �          @���}p�@(��?�=qAM�CE�}p�@Q�?���A�{C�                                    Bxe�g�  �          @���a�@`��?(��@�33C!H�a�@Tz�?���At��C�q                                    Bxe�vZ  �          @��\��Q�@A녽��
�s33C޸��Q�@?\)?   @�ffC:�                                    Bxe�   �          @����
=@0��=��
?s33C��
=@,��?(�@�ffCff                                    Bxe퓦  �          @������@:�H��\)�J=qC�)���@8Q�>��H@��\C�q                                    Bxe��L  �          @�33�c�
@fff�����W�C�3�c�
@e>���@�
=CǮ                                    Bxe���  "          @�=q�L��@s�
>�p�@���C��L��@j�H?��AJ=qC�                                    Bxe���  �          @����J=q@u�?333A   C� �J=q@hQ�?�A�33C
=                                    Bxe��>  �          @��\�E@z�H?k�A'
=C 33�E@j�H?�33A��
C                                    Bxe���  "          @���=p�@\)?k�A'�B���=p�@o\)?�z�A��C Q�                                    Bxe��  
�          @�=q�7
=@��\?@  A��B�{�7
=@w
=?�G�A�B�\                                    Bxe��0  8          @���P��@l�Ϳ:�H�33C\)�P��@qG����
�8Q�C��                                    Bxe��  "          @���S�
@E�?O\)A!C	�S�
@7�?���A���C�                                    Bxe�|  "          @�Q��G
=@\(�?�(�A���C(��G
=@C33@A��C�
                                    Bxe�&"  
Z          @�G��J=q@r�\?G�A\)C���J=q@dz�?�p�A�{C�\                                    Bxe�4�  T          @����{�@Fff>�
=@���C���{�@>{?�  A7�C��                                    Bxe�Cn  �          @�  �aG�@W�?@  A�C@ �aG�@J=q?�\)A�p�C
                                    Bxe�R  "          @��R�Dz�@P  <�>�{C^��Dz�@L(�?#�
A(�C�                                    Bxe�`�  �          @��R�\(�@<(�?�@�RCk��\(�@1�?�\)Ac\)C                                    Bxe�o`  �          @�\)�]p�@��@$z�A��C���]p�?�z�@>�RB=qC                                    Bxe�~  �          @����p�@Q�?��A�(�CJ=��p�@33?��HA��C޸                                    Bxe  
�          @�����{@	��?k�A(��C+���{?�Q�?��Aq��Cn                                    Bxe�R  "          @�Q�����@\)?�  A��\Cp�����@
=q?���A�  C�                                    Bxe��  
�          @����\)@Q�?���As
=C����\)@�?޸RA�=qC��                                    Bxe  �          @������@<(������_\)C�����@<(�>��@:�HC��                                    Bxe��D  "          @���`  @dz�?�R@�\Cu��`  @X��?��
Al(�C��                                    Bxe���  T          @����?\)@u�?��
AlQ�B���?\)@aG�?�p�A��CT{                                    Bxe��  �          @��\���@��
?���AJffB�Ǯ���@��\?�z�A��
B�                                    Bxe��6  T          @�G��.�R@���?���A]p�B�{�.�R@p  ?�Q�A��B�8R                                    Bxe��  T          @���6ff@w�?��HA��B��R�6ff@aG�@
�HȀ\C �                                    Bxe��  T          @����@��@hQ�?�Q�A��C���@��@O\)@A�
=C�
                                    Bxe�(  �          @�Q��a�@B�\?���A���CE�a�@(��@
=A��C.                                    Bxe�-�  T          @�\)�s33@��@   A�33C��s33@�@�A�\)C�{                                    Bxe�<t  �          @�  ���?�=q@A�p�CG����?�33@��A��HC!^�                                    Bxe�K  �          @�
=��{?��?�Q�A�(�C����{?��@\)A���C!��                                    Bxe�Y�  "          @�\)���R?�  ?���A�p�Cff���R?���@  AׅC"+�                                    Bxe�hf  "          @��R���?��H?�A�  C�q���?�=q@
�HA�{C+�                                    Bxe�w  T          @�ff����?޸R?�(�A�33C������?���@G�A���C"
                                    Bxe  "          @�
=��33@��?�z�A�{C!H��33?�z�@33A�\)C
=                                    Bxe�X  T          @�
=��
=@�?�G�A�(�CL���
=?�ff?��A��RC��                                    Bxe��  �          @�\)����@G�?�ffA�G�CO\����?�Q�?���A��\C�3                                    Bxeﱤ  "          @��R�{�@(��?�Q�A�C#��{�@z�?�z�A�  Cu�                                    Bxe��J  �          @�ff���@p�?��A���C� ���@
=q?�=qA��C                                      Bxe���  �          @�{��  @�H?��A��C����  @?��HA�(�Cff                                    Bxe�ݖ  �          @�\)�p  @6ff?�=qA�z�C� �p  @   @A���CB�                                    Bxe��<  �          @�
=����@=q?�\)A�
=C�����@�
@33A��C�
                                    Bxe���  �          @���\)?�  @"�\A�(�C"���\)?@  @.�RB��C)L�                                    Bxe�	�  
�          @�����@
�H?���A��C�f����?��@��A�C                                      Bxe�.  
�          @�ff���?�?���A��
C&f���?�Q�@��Aٙ�C ��                                    Bxe�&�  T          @�{���H?�@�\A�G�C���H?�
=@A�p�C                                     Bxe�5z  
�          @�
=���\?�Q�@�
A��C�����\?\@��A�C��                                    Bxe�D   "          @���c�
@(Q�@\)A֏\C�
�c�
@
=q@,(�Bp�C�                                     Bxe�R�  
�          @�Q��^{@0��@33A�\)C� �^{@�@1G�B{C��                                    Bxe�al  �          @����L(�@S�
?��RA�\)C�3�L(�@8Q�@$z�A���C	�3                                    Bxe�p  
�          @�  �q�@E�?��AR�HC� �q�@4z�?ٙ�A��RCG�                                    Bxe�~�  
�          @��R�n{@L(�?0��A (�CY��n{@@  ?��
Ao
=C
=                                    Bxe��^  
�          @�p��e@Tz�?�\@���C	J=�e@J=q?�\)AR=qC
��                                    Bxe�  T          @�33�O\)@fff>B�\@33C  �O\)@`  ?\(�A$��C��                                    Bxe�  �          @�(��U@a논#�
�\)Ch��U@^{?&ff@�{C�f                                    Bxe�P  T          @����:�H@vff>��
@u�B�L��:�H@n{?��AFffC �                                    Bxe���  �          @���Q�@\��?��A�Q�Cz��Q�@HQ�@ ��A��CJ=                                    Bxe�֜  >          @����e�@[�?L��A�CB��e�@N{?�
=A��C
�                                    Bxe��B  8          @����l(�@0  ?�p�A��CT{�l(�@�@p�A���C�                                     Bxe���  �          @�G��`��@2�\@��Aՙ�C���`��@z�@/\)BG�C�\                                    Bxe��  "          @�G��W�@5�@�A�G�C��W�@ff@4z�B

=C&f                                    Bxe�4  T          @����I��@%@?\)B(�C�
�I��?�p�@[�B*�\C�
                                    Bxe��  �          @����g�@O\)>�G�@�(�C
��g�@Fff?��ADz�C^�                                    Bxe�.�  �          @�\)�w�@>{�����=qC}q�w�@@  >��?��CE                                    Bxe�=&  8          @�Q��z�H@:�H?���AK�CQ��z�H@*=q?У�A�{C�{                                    Bxe�K�  �          @���vff@A�?Q�A�C��vff@5�?�\)A�ffC��                                    Bxe�Zr  �          @�{�w�@�R@�
A�C  �w�?�@(�A��C��                                    Bxe�i  T          @��p  @��@�A���C�
�p  @�@��A�G�C��                                    Bxe�w�  T          @����q�@.�R?�A��
C0��q�@=q?�z�A���CxR                                    Bxe�d  �          @�p��|(�@+�?�A]G�C�|(�@�H?�z�A�C��                                    Bxe�
  �          @�\)�fff@XQ�?+�@�\)C�\�fff@L(�?��Ap��C
n                                    Bxe�  �          @�\)�qG�@E�?z�HA7
=C���qG�@5?�ffA�33C�q                                    Bxe�V  �          @��R�r�\@J=q>\@�(�C:��r�\@A�?xQ�A4  Cc�                                    Bxe���  "          @�{�vff@@  ?#�
@�\)C��vff@4z�?�Q�A`z�C�                                     Bxe�Ϣ  T          @�ff�w�@:=q?uA4(�C���w�@+�?�  A��CB�                                    Bxe��H  T          @����u@Dz�?��A>{CQ��u@4z�?���A�{C��                                    Bxe���  �          @�G��r�\@O\)?:�HACu��r�\@B�\?�=qAt��C5�                                    Bxe���  "          @�Q��qG�@H��?��\A;�
C5��qG�@8��?˅A�Q�C��                                    Bxe�
:  T          @��u�@2�\?���AQG�C�H�u�@"�\?�{A�z�Cz�                                    Bxe��  �          @�p��z=q@0  ?}p�A;\)C���z=q@ ��?��RA�ffCE                                    Bxe�'�  T          @�{�L��@e�>��@�G�C�q�L��@[�?���AY��C��                                    Bxe�6,  "          @�z��R�\@_\)>L��@Q�CO\�R�\@X��?\(�A$(�C&f                                    Bxe�D�  �          @�����@�?��A�  C(���?���?�G�A��C��                                    Bxe�Sx  "          @�z���
=@	��?���AVffC����
=?�33?�G�A�
=C�q                                    Bxe�b  �          @�z��s33@6ff����ffC��s33@9��<�>ǮC�                                    Bxe�p�  �          @�33���@%������RC�
���@"�\>�@�Q�C�                                    Bxe�j  "          @�(�����@�R>�@�p�C�����@ff?^�RA%C޸                                    Bxe�  �          @��\���H@ff=u?@  C:����H@33>�@�
=C�                                    Bxe�  
�          @��H����?�{=�\)?Y��C ������?���>�p�@�ffC!�                                    Bxe�\  "          @��H���?�
=��ff��ffC!H���?�p����
�uC��                                    Bxe�  T          @��
��Q�?��Ϳ�\��=qC ����Q�?�z�L���C�                                     Bxe�Ȩ  
�          @�����
?�(���Q쿋�C�R���
?���>�\)@Y��C�                                    Bxe��N  �          @�����=q@  ����B�\Cz���=q@��>.{@33Cff                                    Bxe���  
�          @����~�R@4z�k��*�HC��~�R@3�
>��R@k�C                                    Bxe���  "          @�33���@$z�?�\@���C�H���@�?}p�A<��C�                                    Bxe�@  
�          @�33��Q�@33?��AICL���Q�?�?�
=A�33C��                                    Bxe��  
�          @��
�z=q@\)?�z�A�p�Cu��z=q@�?�{A���C�)                                    Bxe� �  �          @�z��z=q@$z�?�A���C�R�z=q@  ?��A���C
                                    Bxe�/2  "          @��
��p�@�?^�RA%C�=��p�@�?�ffAx��C�                                    Bxe�=�  T          @�����
@�>�33@�z�C�����
?�(�?=p�Az�C�q                                    Bxe�L~  T          @��
��@ff?Tz�A��C����@	��?�G�Apz�C                                    Bxe�[$  �          @��H���
?�Q�?5A�
C)���
?�G�?���AL��C�                                    Bxe�i�  
�          @��H���@�?��
AE�Ck����@?��HA�z�C                                    Bxe�xp  �          @��\���@33?��AK�C#����?�?�Q�A���Cٚ                                    Bxe�  "          @��\��
=@
=?��AF�HCc���
=?�\)?�
=A�\)C
=                                    Bxe�  S          @��H���\@�\?���Aep�C�R���\@G�?�{A�{C��                                    Bxe�b            @��\��=q@  ?��
Av�HC
��=q?��H?�Q�A�=qCJ=                                    Bxe�  i          @�=q����@G�?���A��HC������?�(�?��A�Q�C�                                    Bxe���  �          @��
���H@(�?��A���C�)���H?��?��A�33CT{                                    Bxe��T  �          @�������@<(�>8Q�@�\C������@6ff?=p�Az�C�q                                    Bxe���  �          @���~{@@�׽�Q�z�HC�\�~{@>{?   @�\)C+�                                    Bxe���  �          @�  �qG�@O\)��������CO\�qG�@P��>u@.�RC+�                                    Bxe��F  �          @�ff�c33@Vff?333A33C�f�c33@J=q?��A{�
C
c�                                    Bxe�
�  T          @�  �b�\@aG�>B�\@��C)�b�\@Z�H?\(�A\)C��                                    Bxe��  �          @�
=�a�@U���\�=p�C�)�a�@^{��33���\Cz�                                    Bxe�(8  �          @����E@c�
��G��nffC���E@o\)�
=q�˅C��                                    Bxe�6�  �          @�33�R�\@]p��G��Q�C�H�R�\@b�\��Q쿏\)C�                                    Bxe�E�  T          @��
�Y��@Z�H��\����C�\�Y��@\��>#�
?�C�                                    Bxe�T*  �          @��
�X��@\(��\)����C�=�X��@_\)=�?���C.                                    Bxe�b�  �          @�33�Tz�@^{�
=��\)C�Tz�@aG�=���?�z�C\)                                    Bxe�qv  �          @���]p�@Q녿L����
C���]p�@W��\)�ٙ�C                                    Bxe�  �          @����fff@E�W
=�%CaH�fff@Dz�>Ǯ@�\)C�=                                    Bxe��  �          @����x��@*�H?W
=A!�C�{�x��@��?��A�(�CǮ                                    Bxe��h  �          @����p  @<(�>8Q�@
�HC��p  @6ff?@  A�C�\                                    Bxe��  "          @�=q��H@dz��=q���
B�L���H@w
=�����Xz�B�8R                                    Bxe���  �          @�z��Q�@j=q�
=q����B�
=�Q�@�Q쿳33��  B�Q�                                    Bxe��Z  �          @�z��'
=@hQ��
=��\)B�z��'
=@{���Q��a��B��                                    Bxe��   T          @��
�#33@aG��
=q�ә�B��#33@xQ쿷
=����B���                                    Bxe��  
�          @��
�3�
@hQ�˅��
=B���3�
@w��Y��� ��B���                                    Bxe��L  T          @��H�(��@u������}B�=q�(��@�Q�
=q��{B��R                                    Bxe��  �          @��\�'
=@e���
=����B���'
=@x�ÿ�Q��d  B���                                    Bxe��  
�          @��\�3�
@e��������HC !H�3�
@tz�\(��$z�B���                                    Bxe�!>  �          @�33�/\)@i����  ��G�B��)�/\)@w��@  �
=B���                                    Bxe�/�  T          @��
�$z�@g
=���R��B���$z�@{����R�l��B�\)                                    Bxe�>�  �          @��
�)��@aG���p���G�B����)��@u���R�n{B�8R                                    Bxe�M0            @�=q�B�\@aG������Y�C��B�\@j�H������
C�\                                    Bxe�[�  �          @�=q�>�R@_\)��Q����Cu��>�R@mp��8Q��
{C                                     Bxe�j|  �          @�Q��,��@b�\��\)��=qB����,��@r�\�c�
�,��B��)                                    Bxe�y"  �          @�Q��P��@Q녿����P(�C�
�P��@[��Ǯ����C�{                                    Bxe���  T          @���W
=@P  �(���   C�3�W
=@S�
    =#�
CaH                                    Bxe��n  �          @�{�o\)@0��=�Q�?��C�)�o\)@,(�?#�
@�(�CW
                                    Bxe��  T          @���s�
@3�
�L�Ϳ(�C���s�
@0��?�\@ƸRC                                    Bxe���  �          @�{��z�@	��>���@���C�=��z�@�\?@  A  C�q                                    Bxe��`  T          @�  ��33?���>�G�@�=qC���33?�p�?L��A�C\)                                    Bxe��  �          @�\)��@(�    <�CW
��@	��>�G�@��
C�{                                    Bxe�߬  T          @�{��33?�(�������Cn��33?�(�>B�\@Q�CxR                                    Bxe��R  T          @�ff���\?�  �B�\��\C�q���\?�  >.{@z�C�R                                    Bxe���  �          @�{�|��@�ÿJ=q�(�C���|��@   �����h��C��                                    Bxe��  T          @�
=���\@(��\(��(��C�����\@z������p�Cn                                    Bxe�D  �          @�ff���?�녿Q�� z�C:����?�\������C�R                                    Bxe�(�  i          @��~�R@�׿h���3�Ch��~�R@����(����C�                                    Bxe�7�  �          @�ff�x��@���Q��j�RC��x��@!G��0����C{                                    Bxe�F6  �          @��R�n�R@#33���
�}C���n�R@0�׿=p����C��                                    Bxe�T�  "          @�
=�\��@8�ÿ�������C\�\��@G
=�E��{C	�R                                    Bxe�c�  �          @���s33@!G���  �u��Cff�s33@.�R�5�
�RCW
                                    Bxe�r(  �          @�
=�p��@   ��{��Q�CQ��p��@.�R�Q�� ��C�                                    Bxe���  
�          @�
=�i��@&ff��Q����RC�\�i��@5�aG��,��C&f                                    Bxe��t  �          @��g�@(�ÿ�������C���g�@7
=�B�\���C                                    Bxe��  �          @�ff�i��@&ff��\)���
C�=�i��@4z�O\)��RCB�                                    Bxe���  T          @����g�@   �����z�Cc��g�@0�׿�  �E��C��                                    Bxe��f  T          @����j�H@{��z����C{�j�H@,�Ϳ^�R�-��C��                                    Bxe��  �          @�p��e�@ �׿�\)���HC�f�e�@2�\�����Tz�C�                                    Bxe�ز  
�          @�p��Q�@3�
��G����C^��Q�@G
=��33�c�Cs3                                    Bxe��X  
�          @�p��K�@;���\���C	\)�K�@N�R�����_33C��                                    Bxe���  �          @���L��@4z��\)����C
�f�L��@I����  �x��C�                                    Bxe��  �          @��H�S33@�R��p�����C{�S33@5������CT{                                    Bxe�J  �          @����S33@p������\)CL��S33@333������C��                                    Bxe�!�  �          @����Tz�@\)�޸R��ffC��Tz�@333��Q��u�C޸                                    Bxe�0�  �          @����P��@,(���=q��p�Cs3�P��@=p��z�H�H��C	�                                     Bxe�?<  T          @����+�@N�R�����
=C��+�@`  �n{�?�B��)                                    Bxe�M�  �          @����>�R@5���=q��  CxR�>�R@I�������w\)Ch�                                    Bxe�\�  �          @�\)�W
=@=q��(����\C^��W
=@-p���
=�t  C�                                    Bxe�k.  �          @��R�A�@%��
=�Σ�Cs3�A�@<(������\)C޸                                    Bxe�y�  �          @���E@7���Q���G�C	{�E@G
=�O\)�'�C�=                                    Bxe��z  �          @��R�H��@0  ��������C
���H��@?\)�W
=�0(�C^�                                    Bxe��   �          @�\)�P  @"�\���H���C�R�P  @2�\�c�
�<  CW
                                    Bxe���            @�
=�6ff@XQ쾳33���\C��6ff@XQ�>�p�@��C
                                    Bxe��l  T          @�
=���@w
=��{���B�8R���@vff>�@�{B�k�                                    Bxe��  �          @����8��@E�����g�C��8��@P  �����ffC�f                                    Bxe�Ѹ  �          @�p��.�R@[����ۅC ���.�R@]p�>W
=@.�RC E                                    Bxe��^  �          @�
=�5@W��(�����C!H�5@[�=�\)?^�RC�)                                    Bxe��  �          @��R�A�@B�\���t��C�)�A�@N{�   �θRC8R                                    Bxe���  �          @�
=�HQ�@=p���Q��vffC�H�HQ�@H�ÿ����C��                                    Bxe�P  �          @�G��(Q�@"�\��H���C��(Q�@@  ������{CG�                                    Bxe��  �          @��R��@ ���B�\�,{C����@Fff�(��p�B��3                                    Bxe�)�  �          @����H@8Q��6ff�!z�B񞸿��H@Z�H�
�H��(�B�G�                                    Bxe�8B  �          @��@333� ���	�C��@QG�������\)B�8R                                    Bxe�F�  �          @��"�\@'��p����C��"�\@E����\)CaH                                    Bxe�U�  �          @�z��p�@P�׿�\)�˅B�=q�p�@e������p  B�B�                                    Bxe�d4  T          @�p��,��@��(���{C�,��@333����
C�3                                    Bxe�r�  T          @��\�P��@!G����R��ffCG��P��@.�R�+��\)C�                                    Bxe���  �          @��\�H��@&ff������CO\�H��@5��=p��33C	�                                    Bxe��&  �          @�=q�\)@�R��H�
G�C
�\)@<�Ϳ�=q�ʸRC.                                    Bxe���  �          @���Dz�@#�
��ff��ffC&f�Dz�@1녿5���C	�=                                    Bxe��r  �          @����X��@Q�.{�z�C޸�X��@{�\)��Q�C޸                                    Bxe��  �          @��
�a�@ff?��HA�33C33�a�?��?��A��C)                                    Bxe�ʾ  �          @���Y��@��?�p�A�  C�=�Y��?��H@	��A��
CB�                                    Bxe��d  �          @���Vff@
�H?���A�=qC��Vff?�(�@  A�
=C��                                    Bxe��
  �          @��R�Q�@  @   A�  C���Q�?�G�@��Bz�C�=                                    Bxe���  �          @���Vff@
�H@ ��A�=qC\�Vff?�@(�BffCn                                    Bxe�V  �          @�z��S�
@�?ٙ�A�  C޸�S�
?�33@
�HA���C(�                                    Bxe��  �          @���>{@��?���A�ffC#��>{?�p�@A��HCL�                                    Bxe�"�  �          @~�R�1�@,��>�ff@�z�Cٚ�1�@!�?��AvffC	��                                    Bxe�1H  �          @����"�\@E���\����C� �"�\@G�>8Q�@'�C+�                                    Bxe�?�  �          @z�H�,��@�?��A�
=C5��,��?�p�?��A�
=C��                                    Bxe�N�  �          @}p��8Q�@p�?��Av{Cz��8Q�@
�H?ǮA�p�C                                    Bxe�]:  �          @�  �Mp�@,��?0��AG�C�3�Mp�@�R?��\A�\)CW
                                    Bxe�k�  �          @��
�K�@!G�?8Q�A (�C��K�@33?�G�A��\C�                                    Bxe�z�  �          @�p��O\)@"�\?&ffA  C��O\)@�?���A�{CO\                                    Bxe��,  �          @���QG�@�R?fffALQ�C�3�QG�?�(�?�\)A�=qC��                                    Bxe���  �          @�=q�G�@{?L��A5�C���G�@�R?��A�p�Cz�                                    Bxe��x  
�          @��
�G�@$z�?G�A/�C�=�G�@�?��A�{CJ=                                    Bxe��  
�          @���B�\@"�\?��
A�
=C\�B�\@(�?���A��C33                                    Bxe���  �          @���S�
@.�R?333A��C���S�
@   ?�ffA�33C�R                                    Bxe��j  �          @�=q�J�H@8Q�>u@I��C	� �J�H@0  ?aG�A=p�C
=                                    Bxe��  �          @��\�B�\@<�;��H���
C���B�\@>�R>B�\@!�C�\                                    Bxe��  T          @�z��R�\@3�
�0���G�C���R�\@8�ý#�
��C
�3                                    Bxe��\  �          @��H�>{@>{���
�\��C���>{@HQ쾨�����
C}q                                    Bxe�  �          @�(��J=q@5�c�
�=C
��J=q@>{�aG��<(�C��                                    Bxe��  �          @��
�AG�@C33�:�H��C���AG�@HQ�#�
�#�
C�                                    Bxe�*N  �          @�z��2�\@P�׿&ff�
�RC���2�\@U�=�G�?�
=C�R                                    Bxe�8�  �          @��\�C33@C33=u?@  C���C33@=p�?@  A ��C�H                                    Bxe�G�  �          @�33�A�@@�׾�  �XQ�C+��A�@?\)>�ff@�
=Cc�                                    Bxe�V@  �          @���>�R@G
=?��@�=qC��>�R@9��?�G�A�Q�CǮ                                    Bxe�d�  �          @�{�P  @>�R>�  @K�C	�\�P  @5?k�AAG�C
�f                                    Bxe�s�  �          @�p��P  @?\)�.{��RC	^��P  @<��?�@ٙ�C	�                                    Bxe��2  �          @��Fff@H�þ��� ��C�3�Fff@E?z�@�C(�                                    Bxe���  �          @��R�N{@E�=u?J=qC=q�N{@?\)?E�A   C	+�                                    Bxe��~  �          @��R�^{@0�׾���W�C���^{@/\)>Ǯ@�  C�                                    Bxe��$  �          @��R�]p�@0��=�G�?�
=Cu��]p�@*=q?@  A  C��                                    Bxe���  �          @�  �Z�H@:=q>�=q@Y��C�)�Z�H@1G�?p��A@z�C                                    Bxe��p  �          @����\��@8Q�?   @��C0��\��@+�?�z�Al��C(�                                    Bxe��  �          @����^{@9��?   @��C{�^{@,��?�Am��C�                                    Bxe��  �          @���S33@E>��@�
=C��S33@8��?�Q�Aq��C
��                                    Bxe��b  �          @����Z�H@9��?\)@��C���Z�H@,(�?�(�A{33C�=                                    Bxe�  �          @�
=�`��@%�?k�A=C���`��@33?�  A�G�C�{                                    Bxe��  �          @�Q��W
=@<��?(�@��HC
���W
=@.{?��A�\)C�                                    Bxe�#T  �          @�Q��[�@5�?.{A33Cp��[�@%?�=qA�C�                                    Bxe�1�  �          @�  �`��@*=q?Y��A.=qC�\�`��@��?���A�p�C�=                                    Bxe�@�  T          @����vff@(�?Tz�A*ffCh��vff?�
=?�=qA�33Cn                                    Bxe�OF  �          @�G��s33@z�?Q�A'�C�)�s33@�
?���A�
=C�
                                    Bxe�]�  T          @����k�@p�?n{A=�C=q�k�@
�H?��RA�  C��                                    Bxe�l�  �          @�  �z�H?�?��AV�RC���z�H?�{?�(�A���C��                                    Bxe�{8  �          @�  �\)?޸R?�{Ae�CxR�\)?�?�  A��C h�                                    Bxe���  �          @���w�@z�?h��A733C{�w�@�?���A�=qCO\                                    Bxe���  �          @��
�s33@�?uA?�CaH�s33@Q�?�G�A�G�C��                                    Bxe��*  �          @��
�l��@&ff?c�
A1p�C��l��@�
?��RA���C\                                    Bxe���  �          @���u@=q?G�AQ�C��u@	��?��A�p�C��                                    Bxe��v  T          @���^�R@7�?n{A:{Cu��^�R@#�
?���A�=qC��                                    Bxe��  �          @��\�{�@33�W
=�'
=C�3�{�@�>�Q�@�\)C�f                                    Bxe���  �          @�=q�xQ�@��=��
?s33CE�xQ�@�
?(��Az�C@                                     Bxe��h  T          @��H�i��@-p�>�@���Ch��i��@!G�?�{A`Q�Ck�                                    Bxe��  �          @���mp�@&ff>�p�@�p�C���mp�@�?�  AJ=qC�                                     Bxe��  �          @�33�n{@'
=>��H@��HC�R�n{@=q?�{A`��C�                                    Bxe�Z  �          @���i��@,(�=���?��C���i��@%�?B�\A{C�q                                    Bxe�+   �          @��\���?��R?uAE�C :����?��H?�ffA��C#                                    Bxe�9�  T          @�z���33?�  ?n{A9G�C#���33?}p�?��HArffC'=q                                    Bxe�HL  T          @����\)?��>B�\@�C{��\)?�Q�?!G�@��\CE                                    Bxe�V�  �          @�(���z�@�>u@?\)C���z�?�33?=p�A�\CQ�                                    Bxe�e�  �          @����=q@ff?�@љ�C�q��=q?�33?��
AN�\C�R                                    Bxe�t>  T          @��
����@(�>�@�ffCff����@ ��?�G�AJ{C�                                     Bxe���  �          @����s�
@p�>k�@8��C(��s�
@�?Tz�A)G�C�)                                    Bxe���  �          @����{�@p�>�(�@�  C�
�{�@�\?xQ�AF�\C��                                    Bxe��0  
�          @�33����@Q�>Ǯ@�CE����?��H?k�A7�
C#�                                    Bxe���  �          @�G��w
=@ff>�?˅C���w
=@\)?5A��C��                                    Bxe��|  �          @����q�@�H�u�AG�CY��q�@��>\@��HC��                                    Bxe��"  
�          @����mp�@��?Y��A-G�C5��mp�@ff?�A�G�C}q                                    Bxe���  �          @�Q��s�
@�
?
=@��CǮ�s�
@?�33Am��CL�                                    Bxe��n  �          @���z�H?�?�  AO33Cٚ�z�H?�{?��HA��C�3                                    Bxe��  �          @�  �}p�?�?aG�A4��C��}p�?��?��A���C�=                                    Bxe��  �          @�Q���=q?���?�@�\)C�\��=q?�z�?z�HAI�C޸                                    Bxe�`  �          @������
?�?   @��HCT{���
?�\)?p��A@Q�C��                                    Bxe�$  �          @�������@G�>���@w
=CT{����?��?O\)A%G�C�q                                    Bxe�2�  �          @������?��>L��@#33C\)���?�\?.{A	��C��                                    Bxe�AR  �          @�Q���  @�
>k�@>{C����  ?�
=?@  A�CE                                    Bxe�O�  �          @�  ����?�z�>���@xQ�CǮ����?�\?G�A ��Cs3                                    Bxe�^�  �          @������?�\)>�z�@j=qC� ���?޸R?B�\A�HC)                                    Bxe�mD  �          @������
?�=q>�{@��
C����
?�
=?O\)A$  C��                                    Bxe�{�  �          @�\)���H?�>��?��HC{���H?��H?�R@�p�CE                                    Bxe���  �          @���{�@	��>L��@!�CT{�{�@G�?@  A��C�                                     Bxe��6  �          @���p  @=q>W
=@,(�CW
�p  @G�?Tz�A*�HCٚ                                    Bxe���  �          @�Q��r�\@=q>k�@:�HC���r�\@G�?Y��A-C(�                                    Bxe���  �          @�{�|��?��R>#�
@
=C5��|��?��?.{A��C�                                    Bxe��(  �          @�ff�x��@Q�>�=q@aG�CY��x��?�p�?Q�A*{C                                    Bxe���  �          @�(�����?��>L��@(��C�����?��
?�RA33CJ=                                    Bxe��t  �          @�33�u@�>u@J�HC+��u?��?B�\A!�C                                    Bxe��  �          @�z��s�
@
�H=���?��CW
�s�
@�
?+�A�C�                                    Bxe���  �          @��tz�@{�#�
��C޸�tz�@��?z�@�\)C��                                    Bxe�f  �          @�\)�O\)@>�R�&ff���C	k��O\)@B�\>#�
@
=C�\                                    Bxe�  �          @�Q��@  @L(����\�R=qC@ �@  @Vff�\)��ffC�H                                    Bxe�+�  T          @�Q��P  @@�׿8Q���C	��P  @Fff=�G�?���CW
                                    Bxe�:X  �          @���a�@,�;���XQ�C���a�@*�H>�@��C�H                                    Bxe�H�  �          @�
=��  ?���=�G�?�33C����  ?���?!G�AG�C�                                    Bxe�W�  �          @��R�w
=?��?��A��HC
�w
=?���?�(�A�=qC J=                                    Bxe�fJ  �          @�  �qG�?��R?�A�p�C:��qG�?��
?�33A�(�C޸                                    Bxe�t�  �          @�G��k�@\)?aG�A2�RC���k�@
�H?\A�=qC�                                    Bxe���  �          @�Q��Y��@3�
?Q�A(��Ck��Y��@\)?�ffA�G�C��                                    Bxe��<  �          @���K�@G
==�G�?���C�H�K�@>{?p��AC\)C��                                    Bxe���  �          @�G��`  @3�
��p���p�C0��`  @3�
>��@�C@                                     Bxe���  �          @����$z�@j=q��\)�dz�B�{�$z�@fff?8Q�Az�B���                                    Bxe��.  �          @�
=�333@[�=��
?�ffC:��333@Q�?�G�AR�\C�                                     Bxe���  �          @�ff�*=q@b�\>#�
@Q�B��f�*=q@W
=?�\)AjffC aH                                    Bxe��z  �          @����W�@*�H?�@��HC���W�@�H?��
A�33CW
                                    Bxe��   �          @����O\)@1G��J=q�(��Cn�O\)@8Q�#�
����C
O\                                    Bxe���  �          @��Vff@7
=<��
>��RCz��Vff@0  ?O\)A(��C�f                                    Bxe�l  �          @���_\)@'�����W
=C)�_\)@%>��@�ffCn                                    Bxe�  �          @����z�H@(�>�  @HQ�C�=�z�H@�\?W
=A,  C��                                    Bxe�$�  �          @����z=q@�ÿ������C^��z=q@{<#�
>#�
Cu�                                    Bxe�3^  �          @����aG�@%�����`z�C� �aG�@1녾�{����C�                                    Bxe�B  �          @����XQ�@(Q쿷
=��p�C{�XQ�@;��#�
�33C�                                    Bxe�P�  �          @�Q��W�@0  �����c�C�{�W�@<(������s�
C
޸                                    Bxe�_P  �          @����Q�@G��8Q��G�Cn�Q�@C33?333A�C	)                                    Bxe�m�  �          @�G��O\)@J�H=u?8Q�C�H�O\)@A�?p��A>�HC�                                    Bxe�|�  �          @�G��I��@N�R=���?��
CG��I��@Dz�?�  ALz�C��                                    Bxe��B  �          @����\(�@0  ?k�A=p�CJ=�\(�@��?�z�A��HC:�                                    Bxe���  �          @�  �]p�@&ff?�{Af{C��]p�@(�?�ffA�\)C                                    Bxe���  �          @�Q��Z�H@1G�?�@��HC�q�Z�H@   ?���A��RC�
                                    Bxe��4  T          @��R�]p�@0�׾�  �L��Cn�]p�@.{?
=q@߮Cٚ                                    Bxe���  �          @�\)��=q@\)?�G�ATz�B�B���=q@b�\@ffA�33B�#�                                    Bxe�Ԁ  �          @����I��@J=q?z�@�C���I��@7�?���A�{C	�                                    Bxe��&  �          @����=p�@5?c�
AC�
C8R�=p�@{?�z�A�{C.                                    Bxe���  �          @�\)��{@�G�?uAFffBߔ{��{@fff@�
A��HB�L�                                    Bxf  r  �          @��R��p�@c33?�\)Aʏ\B�{��p�@7�@333B�B�B�                                    Bxf             @��R�@  @(Q�?�z�A��C
��@  ?�(�@%�B��C��                                    Bxf �  �          @���:=q@AG�?�
=A���C��:=q@�R@�RA�z�C�                                     Bxf ,d  �          @�Q��P  @0  ?�A���C���P  @�R@	��A�Q�C��                                    Bxf ;
  �          @�  �~{?�
=?\(�A0  C��~{?���?���A���C��                                    Bxf I�  �          @����qG�@�
?aG�A3\)C�=�qG�?��H?�  A��
C��                                    Bxf XV  �          @�G��]p�@7�>�
=@���CO\�]p�@(Q�?�p�A|  C                                    Bxf f�  T          @���Z�H@7��L���!G�C��Z�H@333?#�
A  C��                                    Bxf u�  �          @���\(�@=p���33���
C\)�\(�@;�?�@��
C�)                                    Bxf �H  �          @�G��g�@*�H�����C���g�@,��>�=q@[�CE                                    Bxf ��  �          @����c�
@2�\=��
?xQ�C޸�c�
@)��?c�
A4��CQ�                                    Bxf ��  �          @����c33@2�\>�?�
=C�=�c33@(Q�?p��A@Q�Ck�                                    Bxf �:  �          @���tz�@(��Y���/\)C33�tz�@�8Q��  C}q                                    Bxf ��  �          @����^{@0  �8Q���
C�)�^{@5=�?�p�C�3                                    Bxf ͆  �          @�\)�P��@AG�>�\)@h��C	(��P��@3�
?�z�Ap��CE                                    Bxf �,  �          @���^{@&ff�����z=qC0��^{@$z�>�@�33Cz�                                    Bxf ��  �          @�\)�\)?�Q�
=���HC��\)@G�<#�
=��
C
                                    Bxf �x  �          @�
=�{�@ff�k��9��C��{�@z�>���@�{C:�                                    Bxf  �          @��tz�@	���k��C�
C���tz�@�>��@���C��                                    Bxf�  �          @�ff�i��@�>�ff@�
=C���i��@?�\)Ao
=C@                                     Bxf%j  T          @�
=�y��?�
=�G��#\)C���y��@��.{�  C�3                                    Bxf4  �          @����Q�?�33������ffC�)��Q�?�Q�>8Q�@C5�                                    BxfB�  �          @����u@��\)�f�\Ch��u@��ff��{C�R                                    BxfQ\  �          @����z=q?�\)������RCs3�z=q@
�H�+����C�R                                    Bxf`  �          @�\)��z�?��\��=q�_33C"���z�?���&ff���C�f                                    Bxfn�  �          @�p��|(�?�Q쿁G��\Q�C#=q�|(�?�Q�(���C�q                                    Bxf}N  �          @�ff�j�H@�H�����  C��j�H@�R>#�
@G�C�                                    Bxf��  �          @�ff�[�@.{�������C�{�[�@.{>�(�@��C��                                    Bxf��  �          @���AG�@<�Ϳ�p���=qC�H�AG�@L(���\)�l(�Cp�                                    Bxf�@  �          @�z��G
=@%��������CJ=�G
=@=p��G��$��Cn                                    Bxf��  �          @�(��G
=@#�
�������RC���G
=@8Q�(���RC	@                                     Bxfƌ  �          @���7�@3�
��{��z�C�)�7�@J=q�.{��
C33                                    Bxf�2  �          @�(��C�
@'�����{C}q�C�
@@  �J=q�'�C�{                                    Bxf��  �          @�33�W�@33������z�C�q�W�@(��fff�BffC!H                                    Bxf�~  �          @�����@l�ͿL���)��B����@qG�>�p�@��\B�.                                    Bxf$  �          @�p��J=q@#�
��{����C���J=q@<(��@  �G�C	{                                    Bxf�  �          @�{�C33@8�ÿ�G����C�)�C33@HQ쾞�R���CB�                                    Bxfp  �          @�\)�#�
@b�\?+�Az�B���#�
@J=q?�p�A��RC ��                                    Bxf-  �          @��*�H@`��=�?�ffB���*�H@S33?���A|  C ��                                    Bxf;�  �          @�p��5�@Vff��Q���ffC&f�5�@S33?333AG�C��                                    BxfJb  �          @����@j=q�8Q����B�.��@b�\?uAIB�                                      BxfY  �          @�{�,(�@`��>B�\@ ��B���,(�@Q�?��
A�  Cc�                                    Bxfg�  �          @�
=�5�@P�׿
=��=qC�3�5�@R�\>�ff@��RC�                                     BxfvT  �          @�ff�Z�H@�Ϳ�\)���HCY��Z�H@0�׿
=q��G�C
                                    Bxf��  �          @�{�p�@=p��Q���C���p�@^�R�����g33B��                                     Bxf��  �          @��R��z�@o\)����G�B�uÿ�z�@��Ϳ&ff�33Bٞ�                                    Bxf�F  T          @�
=��@P���  ��B�B���@s33��{�f�\B�                                    Bxf��  �          @��R��\@O\)����  B��{��\@r�\����m�B�R                                    Bxf��  �          @�{��z�@HQ�����  B�׿�z�@p  ������B��                                    Bxf�8  �          @�{�QG�@<(���G�����C	���QG�@;�?�\@ӅC
{                                    Bxf��  �          @�ff�C�
@>�R�����aC�\�C�
@J�H���
�z�HC�                                    Bxf�  �          @���"�\@
=�"�\��
C	��"�\@B�\��z����C�)                                    Bxf�*  �          @����{@
=q�8Q��%=qC
�=�{@>{�33�߅C                                    Bxf�  �          @�(��?�\)�W
=�HC
.�@5�%�33B���                                    Bxfv  �          @�z����@;�����(�CW
���@\(����
�]B���                                    Bxf&  �          @��
�Q�@7��
=��C�q�Q�@X�ÿ�=q�iG�B�8R                                    Bxf4�  �          @�z��Dz�@.�R���H��z�C
L��Dz�@C�
��\���C�                                    BxfCh  �          @�p��X��@Q쿾�R��{C���X��@.�R�#�
��HC(�                                    BxfR  �          @�z��`  @�\���\�ZffC�H�`  @\)�k��@��C�
                                    Bxf`�  �          @����C33@A녿fff�=��C=q�C33@J=q=�?���C
=                                    BxfoZ  T          @����S�
@5�����{Ch��S�
@6ff>��@�p�C8R                                    Bxf~   �          @����c33@"�\��p���=qCaH�c33@!G�>�@�\)C��                                    Bxf��  �          @�z��W�@(��?333A(�C�f�W�@G�?\A��
C�                                    Bxf�L  �          @��@��@J�H�#�
��\C���@��@C33?aG�A:�RC�H                                    Bxf��  �          @�z���
@Z=q�������
B�L���
@i������\B��                                    Bxf��  �          @�(���@b�\��=q��  B�=q��@q녽��Ϳ�=qB��                                    Bxf�>  �          @�(���\@hQ�Q��,��B�z���\@l(�>��@�{B�                                    Bxf��  �          @����33@g
=�fff�>�HB����33@mp�>���@��\B�                                    Bxf�  �          @�p��   @e��=q���B�33�   @z=q���R����B�(�                                    Bxf�0  �          @��33@e��c�
�=��B�p��33@j�H>�{@�B��                                    Bxf�  �          @��333@Y��>�  @Q�Cs3�333@HQ�?�{A��C�{                                    Bxf|  �          @����+�@]p���Q쿘Q�B�u��+�@S33?��A`  C
                                    Bxf"  �          @���<��@P  >.{@(�C8R�<��@@��?��RA���Ch�                                    Bxf-�  �          @�(��/\)@X�ü#�
��G�C �R�/\)@L��?�\)Am��C��                                    Bxf<n  �          @����E�@B�\?(�A ��CW
�E�@*�H?˅A���C)                                    BxfK  �          @�z��Dz�@@  ?Y��A2�HC���Dz�@"�\?�ffA�=qC\)                                    BxfY�  �          @����4z�@L(��#�
�\)Cp��4z�@AG�?��
A_33C                                      Bxfh`  T          @��\�5�@E>B�\@%�C�=�5�@6ff?��HA�=qC޸                                    Bxfw  �          @�(��-p�@Y��>k�@Dz�C �)�-p�@G�?�{A�p�C��                                    Bxf��  �          @�z��C�
@E>\@���C��C�
@1�?�33A�\)C	Ǯ                                    Bxf�R  �          @��
�fff@�\?h��AA�C���fff?���?У�A�p�C��                                    Bxf��  �          @��H�n{?�ff?��RA��\Ck��n{?s33?�
=A��C%��                                    Bxf��  T          @���p��?J=q@33A���C(\�p��=L��@��A���C3.                                    Bxf�D  �          @��
�s�
?h��?�A�33C&���s�
>k�@�\A���C0�                                    Bxf��  �          @�33�z�H?���?�Q�A�33C#���z�H?�?�p�A�
=C+��                                    Bxfݐ  �          @����a�@(�=#�
?�CO\�a�@�?^�RA:�RC�                                    Bxf�6  �          @����(��@_\)�����~�RB�=q�(��@X��?fffA=�B��                                    Bxf��  �          @�p���@w
=�fff�>�HB�LͿ�@{�>��@�{B�u�                                    Bxf	�  �          @���G�@r�\�5�B����G�@s33?!G�Ap�B��                                    Bxf(  �          @�=q��@s�
�O\)�$Q�B�R��@w
=?
=q@�=qB�{                                    Bxf&�  �          @�=q��@�(���R���B�녿�@��H?W
=A(��B�Q�                                    Bxf5t  �          @�녿�Q�@���B�\�B�uÿ�Q�@z�H?��RA|��B�\                                    BxfD  �          @��H� ��@u�u�=p�B�W
� ��@g�?��\A�=qB�aH                                    BxfR�  �          @����
@��
=�Q�?���B�\��
@u�?�  A�ffB�ff                                    Bxfaf  �          @�녿�
=@�33>�(�@�ffB�z��
=@l��?�=qA��HB�33                                    Bxfp  �          @�����\@qG�?��@���B�=��\@U�?�\)A���B�\                                    Bxf~�  �          @���H��@J�H<#�
>��C�\�H��@=p�?�{AfffC��                                    Bxf�X  �          @����a�@1녾�G���z�C�\�a�@0��?��@�  C                                      Bxf��  �          @�=q�u�@(�����Q�C}q�u�@{>�Q�@�=qC33                                    Bxf��  �          @���k�@
=q��\)��ffC���k�@ �׿��أ�C�R                                    Bxf�J  �          @�  �x��?˅������{C���x��@�\�k��<z�CL�                                    Bxf��  �          @���r�\?��ÿ�Q���{C �
�r�\?�zῬ����\)C:�                                    Bxf֖  �          @�{�h��?��R��p��ծC���h��@��=q��z�C
                                    Bxf�<  �          @������@mp�?��RA���B����@1�@J�HB2  B���                                    Bxf��  �          @�G����R@u�?�33A�=qB�\)���R@:�H@HQ�B.�RB��                                    Bxf�  �          @�  ��z�@���?�=qA��B�Q쿔z�@L��@9��B!�\B���                                    Bxf.  �          @��R��p�@z�H?��HA�  B�\)��p�@I��@0  B�B�W
                                    Bxf�  �          @��R���@u?�33A�  B��Ϳ��@:�H@H��B5G�B��)                                    Bxf.z  T          @��R�  @j=q?E�A!B�B��  @I��@G�A�  B�33                                    Bxf=   �          @��g�@!녾����z�C���g�@�H?B�\A\)C0�                                    BxfK�  �          @�{�b�\@*=q=�G�?�C��b�\@��?�ffA[�
CL�                                    BxfZl  �          @�{�'
=@`��?�@�{B�G��'
=@E?�  A�G�C5�                                    Bxfi  �          @����*=q@XQ�?�@�33C 5��*=q@>�R?�Q�A��
Cٚ                                    Bxfw�  �          @���&ff@a녾.{���B��)�&ff@Vff?���Ah��B���                                    Bxf�^  �          @�p��0  @[�=�G�?���C ���0  @J�H?�=qA��HC�                                    Bxf�  �          @�ff�>�R@QG�>�33@�  CE�>�R@;�?�  A�Q�Cs3                                    Bxf��  �          @�\)�9��@QG�?k�A>=qC�{�9��@.{@�\A�
=C�)                                    Bxf�P  �          @�Q��E@N�R>��H@ƸRC�q�E@5?�\)A��HC	u�                                    Bxf��  �          @����3�
@Z=q?Y��A.�\Cz��3�
@7�@�A�=qC^�                                    BxfϜ  �          @����,��@_\)?}p�AJffB�u��,��@8��@(�A�p�C                                    Bxf�B  �          @�G����@j�H?���Aap�B�p����@AG�@
=A�p�C ��                                    Bxf��  �          @��R��  @vff?��A���Bހ ��  @>�R@;�B%  B�8R                                    Bxf��  �          @�\)��z�@s33?���A�=qB�Q��z�@C33@'
=B�B��                                    Bxf
4  �          @�\)�@tz�?�G�APz�B�8R�@L(�@�A�B�Q�                                    Bxf�  �          @�\)�33@qG�?��\A��
B�.�33@C33@#�
B�B�                                    Bxf'�  �          @����
�H@u�?n{A?�B�8R�
�H@N{@G�A�ffB�{                                    Bxf6&  �          @�Q���@l(�?�G�AO\)B��{��@C�
@�\A��HB���                                    BxfD�  �          @����\)@j�H?L��A#�
B�\)�\)@G�@A�p�C ��                                    BxfSr  �          @�
=�@w�?(�@�B�3�@W�?��RAՅB�                                    Bxfb  �          @�ff�   @w�?W
=A/33B���   @Q�@p�A��B�                                    Bxfp�  �          @�����H@���?�R@�
=BӅ���H@o\)@
=qA�  B׽q                                    Bxfd  �          @�
=���@\)?�@�33B�p����@`��?�(�Aҏ\B�Q�                                    Bxf�
  �          @�����@���>�ff@�Q�B�=��@dz�?�z�A�  B�B�                                    Bxf��  �          @���(�@qG�?:�HA��B����(�@N�R@�A��B�(�                                    Bxf�V  �          @����z�@q�?(�@�z�B����z�@Q�?�p�A�B�p�                                    Bxf��  T          @�=q��{@��\>��
@���B��
��{@y��?�z�AƏ\B�k�                                    BxfȢ  �          @�녿�G�@���>�
=@��B��ÿ�G�@s33?��RA�\)B�W
                                    Bxf�H  T          @�G����@�p�?&ffA��B�����@g�@
=qA�\B�                                    Bxf��  �          @����
=q@{�>�
=@��B����
=q@^�R?�{A�B��                                    Bxf��  �          @�Q��=q@�p�>�
=@�p�B�k���=q@l��?��HA�\)B��                                    Bxf:  �          @�\)�u@�33>��R@|(�B̽q�u@z�H?�A��
B�ff                                    Bxf�  �          @����8Q�@�\)>�@��
B�z�8Q�@~{@Q�A��
B�\                                    Bxf �  �          @�녽�\)@�Q�>�@�(�B�����\)@�  @	��A�Q�B��H                                    Bxf/,  �          @�=q���@�\)?
=q@�G�B�{���@|(�@(�A��HB��3                                    Bxf=�  �          @�=q��@�Q�>\@�\)B�(���@�G�@�
A�z�B³3                                    BxfLx  �          @��\���@�  >�Q�@�=qB�33���@���@�\A�33B���                                    Bxf[  �          @�녿@  @�\)>8Q�@�\B���@  @��\?�\)A�=qB���                                    Bxfi�  �          @�G����H@��>��@S�
B�����H@���?���A�=qB��)                                    Bxfxj  �          @�Q��\@�(��W
=�-p�B�uÿ�\@y��?���A���B��f                                    Bxf�  �          @�(���Q�@�  ��Q쿕B��)��Q�@n�R?���A�p�B���                                    Bxf��  �          @�p����@��\���
���B��f���@s33?�p�A���B�Ǯ                                    Bxf�\  �          @������@�Q���Ϳ�ffB�����@o\)?�Q�A��B���                                    Bxf�  �          @��R��33@���>W
=@0  B�{��33@hQ�?޸RA�{B�=q                                    Bxf��  �          @��׿�(�@��>W
=@(Q�B�Ǯ��(�@j�H?�  A�z�B�z�                                    Bxf�N  �          @��׿�@���>��@��B���@e�?�
=A˅B�Q�                                    Bxf��  �          @��ÿ�z�@���?J=qA z�B��ÿ�z�@Y��@G�A��
B�R                                    Bxf�  �          @��ÿ�G�@���?k�A<��B�녿�G�@XQ�@=qB�B���                                    Bxf�@  �          @��R��z�@w
=?h��A?�B�{��z�@Mp�@B 33B�z�                                    Bxf	
�  �          @�z��#�
@\(�?
=q@�(�B�\)�#�
@=p�?�=qA�{Cٚ                                    Bxf	�  �          @���'
=@Vff�(���ffB��)�'
=@U?0��A33B��                                    Bxf	(2  �          @��H��@I������p�B�p���@i�������B�                                    Bxf	6�  �          @�{����@N�R�ٙ���=qB�.����@h�þ��R��G�B�W
                                    Bxf	E~  �          @����\@G���G����HB�G���\@c�
�����ffB�                                    Bxf	T$  �          @�(�� ��@Q��  �ffC��� ��@Dzῑ����HCB�                                    Bxf	b�  �          @�33�9��?���   �\)C@ �9��@p����
��(�C��                                    Bxf	qp  �          @����@  ?��\����(�C!5��@  ?�׿�ff��C��                                    Bxf	�  �          @�  ?�@�녾aG��;�B��R?�@tz�?��A��B���                                    Bxf	��  �          @���?�G�@���>\)?�=qB���?�G�@h��?ٙ�A�G�B��                                    Bxf	�b  �          @��H?p��@��������B�{?p��@z=q?��A��B�\                                    Bxf	�  �          @��H?W
=@�\)��z��y��B��\?W
=@�Q�?�33A���B�Q�                                    Bxf	��  �          @���>��H@�=q��ff��ffB���>��H@���?��A�\)B�p�                                    Bxf	�T  �          @�z�?��\@������=qB��?��\@|(�?���A�ffB���                                    Bxf	��  �          @�ff?G�@�33����\)B���?G�@���?�{A��RB�L�                                    Bxf	�  �          @�  ?��
@��H��G���=qB�ff?��
@��?���A�\)B�B�                                    Bxf	�F  �          @�?�z�@���\��Q�B��?�z�@~{?�ffA�33B��)                                    Bxf
�  �          @�(�?���@�녾�(����
B��H?���@y��?�p�A�  B�B�                                    Bxf
�  �          @�?��
@|�Ϳh���>�\B�k�?��
@\)?=p�A�B���                                    Bxf
!8  �          @�{@G�@i����{�g�
Bh33@G�@r�\>�G�@��Bl
=                                    Bxf
/�  �          @�?�
=@�  ���H�}G�B��{?�
=@���?   @�{B��H                                    Bxf
>�  �          @���?�\@q녿����{B�?�\@�  >�z�@q�B�33                                    Bxf
M*  �          @���?�ff@z=q�=p��z�B���?�ff@xQ�?c�
A;�
B�L�                                    Bxf
[�  �          @�33?�Q�@����{��G�B�(�?�Q�@|��?�{A��B��=                                    Bxf
jv  �          @�(�?�G�@�{�.{��B���?�G�@��?��Aa��B�u�                                    Bxf
y  �          @�p�?��@�(��}p��QB��?��@�?B�\A�HB�p�                                    Bxf
��  T          @��
>�@��ÿ!G���B�8R>�@�p�?�33AuG�B��f                                    Bxf
�h  �          @���>�z�@�  �s33�H��B�L�>�z�@���?Y��A4  B�W
                                    Bxf
�  �          @�p�?W
=@������m�B�\)?W
=@���?(��A	�B��H                                    Bxf
��  T          @�z�>�33@��
��Q����B�G�>�33@�33>�Q�@�ffB�Ǯ                                    Bxf
�Z  �          @��;�33@�(���������B�����33@��>��@�p�B�B�                                    Bxf
�   �          @��>�Q�@�p���Q��}B�.>�Q�@���?(�A ��B�u�                                    Bxf
ߦ  �          @��R?�p�@h�ÿ������Bu�R?�p�@}p�=L��?.{B}�                                    Bxf
�L  �          @�
=?�R@��ÿu�S33B�k�?�R@�=q?E�A)p�B��{                                    Bxf
��  �          @��׾�@�  ��{�i��B��{��@��\?8Q�A�
B��                                    Bxf�  �          @��\?Y��@��\��  �K�
B��R?Y��@��?\(�A/\)B��H                                    Bxf>  �          @��>��@�{������  B��>��@��H?(�@�B�Q�                                    Bxf(�  "          @���?Tz�@��
�ٙ���33B�k�?Tz�@�>u@:�HB��
                                    Bxf7�  �          @��׾��?�33��\)�3Bȳ3���@AG��Q��=(�B��R                                    BxfF0  �          @���>u@��tz��hffB�W
>u@n�R� �����B��{                                    BxfT�  �          @��\?�@_\)�#�
��B���?�@�\)�c�
�=G�B���                                    Bxfc|  T          @��ÿ���@�=q=L��?�RB�zῙ��@���?�
=A�
=B�aH                                    Bxfr"  �          @��ÿ���@�{?^�RA(��B�G�����@k�@'�B(�B�{                                    Bxf��  �          @�Q쿇�@�=q>�ff@��B�33���@~{@33A�B���                                    Bxf�n  �          @�
=�fff@�=q�(��� Q�B�8R�fff@�?�ffA�
B��f                                    Bxf�  �          @��R���
@��ÿ#�
��=qB͞����
@�z�?�ffA��\B�k�                                    Bxf��  �          @���}p�@�\)�=p��33B�  �}p�@�(�?�Q�AlQ�B͊=                                    Bxf�`  �          @�=q�k�@��������Z{B�G��k�@�33?Q�A&�RB���                                    Bxf�  �          @��׿��@w
=��������B�\���@��>���@�Q�BУ�                                    Bxfج  �          @������H?޸R�mp��b�\C�
���H@G��(���G�B�u�                                    Bxf�R  �          @�=q��  @�=q�\)��p�B�k���  @��R?��
A�G�B�G�                                    Bxf��  �          @��R����@��R�����B�p�����@�  ?�
=A��
B�                                    Bxf�  T          @����{@�=q���H�ÅB��H��{@�(�?�{A��\Bޙ�                                    BxfD  �          @�z���@�Q�p���<  B��Ϳ��@�Q�?n{A9B�Ǯ                                    Bxf!�  �          @�G��z�@y���0���p�B����z�@u�?�G�AQB��)                                    Bxf0�  �          @����
�H@p  ����aG�B�.�
�H@w
=?
=@��HB�                                    Bxf?6  �          @�
=�Ǯ@���.{�  B݊=�Ǯ@s33?���A�B���                                    BxfM�  �          @�  ��@~{�u�5B�8R��@g�?�\)A��B�                                    Bxf\�  �          @����AG�@:=q?h��AD(�C�q�AG�@��@�
A�
=C33                                    Bxfk(  �          @�\)�K�@,(�?�{A���C�\�K�?�G�@%B  C
                                    Bxfy�  �          @�\)�E@7
=?\A���C	B��E?���@%B33C��                                    Bxf�t  �          @��\��@,(�?�\@�p�C\�\��@{?���A���C@                                     Bxf�  �          @�
=�c�
@%�?#�
A�C��c�
@�
?�
=A��C��                                    Bxf��  �          @�
=�P  @@  ?
=q@�{C	E�P  @\)?�G�A�ffC��                                    Bxf�f  �          @�
=�G�@H��?\)@�p�C�G�@'
=?�A���C
                                    Bxf�  �          @�
=�AG�@QG�>B�\@�HC�3�AG�@8Q�?ǮA�ffC\)                                    BxfѲ  �          @���0��@Z=q��=q�a�C ���0��@L��?��HA�C�                                    Bxf�X  �          @�ff�<��@S33=L��?(��C���<��@=p�?���A���C��                                    Bxf��  �          @��R�=p�@Tz�=#�
?�C���=p�@?\)?��HA�G�CǮ                                    Bxf��  �          @��R�@  @QG����
����C���@  @?\)?���A��HC!H                                    BxfJ  �          @�=q�Y��@�
?�ffA��RC���Y��?��
@
�HA�C�=                                    Bxf�  �          @��c33?�
=?�Q�A�\)C� �c33?��\@ffB��C$                                    Bxf)�  �          @�
=�W�?�@�\A�p�C���W�?&ff@6ffB��C)�                                    Bxf8<  �          @��E�?��
@(Q�B��C  �E�>��@HQ�B4�HC+B�                                    BxfF�  �          @�
=�H��?\@1G�B��C)�H��>8Q�@J=qB4p�C0�3                                    BxfU�  T          @����Dz�?˅@:�HB C���Dz�>.{@Tz�B<�C0�\                                    Bxfd.  �          @�Q��A�?�G�@4z�BffC޸�A�>�p�@S33B=�C-
                                    Bxfr�  T          @�Q��5�@ ��@5B(�C���5�?
=@\(�BG��C(T{                                    Bxf�z  �          @�Q��?\)@�\@,(�Bz�C��?\)?+�@S33B<��C'^�                                    Bxf�   �          @����K�?���@)��B=qC(��K�?   @K�B2p�C+�                                    Bxf��  �          @����<��?�@7�B��Cc��<��>Ǯ@W�BBffC,}q                                    Bxf�l  �          @�G��{?��R@_\)BI�C���{�8Q�@r�\Bc�C8!H                                    Bxf�  �          @����2�\?���@=p�B#�
C\�2�\>��@`��BLp�C*p�                                    Bxfʸ  �          @�=q�E@
�H@#�
B�RC�{�E?W
=@P  B6  C$Ǯ                                    Bxf�^  �          @���R�\@*�H?��
A�  C��R�\?У�@1G�B�HC�                                    Bxf�  �          @���Q�@8��?���A���C
�f�Q�@ ��@{B��C��                                    Bxf��  �          @�=q�C�
@)��@�AٮC)�C�
?�(�@A�B&��CJ=                                    BxfP  �          @�33�Q�@E�?G�AG�C� �Q�@�@33AՅCp�                                    Bxf�  �          @���<(�@^�R>��?�C(��<(�@C�
?�z�A�33C��                                    Bxf"�  �          @�  �AG�@S�
�Ǯ��  CaH�AG�@H��?���Ab�\C�{                                    Bxf1B  �          @�\)�У�?��R@G
=BZQ�C	�3�У׼�@\��B�=qC5                                    Bxf?�  �          @��Ϳ�(�?��@�p�B��C	�Ϳ�(��!G�@��\B��=CK)                                    BxfN�  �          @��\�h��=��
@�{B��qC/#׿h�ÿ��@���By��Ct:�                                    Bxf]4  �          @�
=�!G�>�G�@��
B�{C
�!G����@��HB��fCx�R                                    Bxfk�  �          @�Q�Tz�?���@�  B}�
B��Tz��@�B���C<k�                                    Bxfz�  �          @��H��Q�@�H@�Q�B`33B�{��Q�>���@��B�C!��                                    Bxf�&  �          @��
�\(�@8Q�@s�
BO�B�8R�\(�?c�
@�{B�k�C�{                                    Bxf��  �          @�녿
=@J�H@c�
B?\)B���
=?��\@��HB�8RB�                                    Bxf�r  �          @�Q쿘Q�@  @���Bf�\B�\��Q�>u@�33B��C(��                                    Bxf�  �          @����  @*�H@dz�BE�\B�p���  ?O\)@�z�B���C��                                    Bxfþ  �          @���ff@=q@|(�Bh��B�L;�ff>���@�33B���C
B�                                    Bxf�d  �          @��R�8Q�@Vff@N{B,�
B�Q�8Q�?���@��
B��B�k�                                    Bxf�
  �          @�Q�5@s33@0��B�B�8R�5@
�H@�p�Bt�B�B�                                    Bxf�  �          @�Q�J=q@~�R@p�A�Bʀ �J=q@{@~�RBc��Bי�                                    Bxf�V  �          @���(�@�z�@��A�=qB�녿(�@.{@tz�BWp�B̀                                     Bxf�  �          @��R=L��@��?�{A�ffB��{=L��@A�@eBGQ�B���                                    Bxf�  �          @�z�=L��@�33?��A�B�Ǯ=L��@Mp�@Tz�B8(�B�W
                                    Bxf*H  �          @��
�s33@e@Q�A�\)Bѳ3�s33@  @`��B\B���                                    Bxf8�  �          @�33�  @J�H@Q�A��B��\�  ?�\@c33BL��C�                                     BxfG�  �          @�z��Q�@[�?�(�A�{B�ff�Q�@�@S33B6�C	k�                                    BxfV:  �          @��Ϳp��@S33@G
=B(�
B���p��?�ff@�Q�B��HB�Ǯ                                    Bxfd�  �          @�\)��
=@n�R@   A�{B�3��
=@�@]p�B@��C p�                                    Bxfs�  �          @�(���G�@dz�@\)A��B螸��G�@�@g
=BP��C                                    Bxf�,  �          @��H��G�@6ff@[�BBp�B�(���G�?�  @��HB�Cp�                                    Bxf��  �          @����z�@�@tz�Be��B�=q�z�>���@�\)B�G�C}q                                    Bxf�x  �          @��ÿ�33@X��@'�B  B��ÿ�33?�@w�BlG�B��=                                    Bxf�  �          @�z���@b�\@A�(�B噚���@ff@l(�BX�HB��H                                    Bxf��  �          @�ff����@e@%BG�B�𤿹��@�@|(�Bf��B�
=                                    Bxf�j  �          @�ff��z�@���?�A�\)B�B���z�@2�\@\��B?\)B�p�                                    Bxf�  �          @�=q����@�{?��HA��B�  ����@E�@L��B'G�B�L�                                    Bxf�  �          @�
=��33@�z�?��HAm�B�(���33@H��@=p�Bp�B�.                                    Bxf�\  �          @�{� ��@s�
?fffA2ffB���� ��@?\)@!�B  C�q                                    Bxf  T          @�{��ff?�=q?Y��A(Q�C}q��ff?��R?˅A�p�C#�=                                    Bxf�  �          @�
=��  @G���G���=qC�{��  ?���?Q�A ��C�                                     Bxf#N  �          @�ff����?У׿��\�HQ�C.����?�z�#�
�   C�                                    Bxf1�  �          @���xQ�@
=��������C� �xQ�@!녾�������Cٚ                                    Bxf@�  T          @��
�l��@������G�Ck��l��@,�ͽ�Q쿐��Cٚ                                    BxfO@  T          @��G�@7
=�������HC	���G�@L��<��
>�  CE                                    Bxf]�  �          @�ff�#33@z=q�������
B�B��#33@j=q?�
=A��\B�Ǯ                                    Bxfl�  �          @�p��   @z=q��{���B�33�   @�33?z�@��B�\                                    Bxf{2  �          @��
��@�G��W
=�(��B�p���@~{?���A\(�B�(�                                    Bxf��  �          @��H��@{��h���8Q�B���@z=q?}p�AIG�B�                                      Bxf�~  �          @�  ��p�@`�׿�33��(�B��Ϳ�p�@\)���
��G�B�R                                    Bxf�$  �          @�����
@��\���H���
B�B����
@u?�Q�A��B߅                                    Bxf��  �          @���   @|(�������
B��)�   @n{?��A��B왚                                    Bxf�p  A          @�ff��33@�Q�>\)?�{B晚��33@_\)?�(�A�33B���                                    Bxf�  �          @��\��=q@|��>�@�(�Bߞ���=q@R�\@\)A��RB�\)                                    Bxf�  �          @�p���@s�
?�Q�A���B�\��@5�@3�
B*33B�Ǯ                                    Bxf�b  �          @�z�^�R@qG�?�
=A�{B�  �^�R@,(�@@��B;=qB��                                    Bxf�  �          @�ff�^�R@w
=?�=qA�z�B�k��^�R@3�
@=p�B4B�p�                                    Bxf�  �          @�G���@|��?�33Az{B��H��@>{@6ffB'
=B��)                                    BxfT  �          @�G���p�@vff?�\)Ar�RB�.��p�@8��@1G�B!��B�L�                                    Bxf*�  �          @�p���{@y��?k�A@��B����{@A�@'�B�RB��                                    Bxf9�  �          @�\)��  @|��?�33A�(�B�z��  @6ff@Dz�B.\)B�Q�                                    BxfHF  �          @��R�(��@�(��W
=�.�RB��(��@~�R?���AĸRBƸR                                    BxfV�  �          @������@�
=?�\@���B��\����@n{@!�Bz�B�B�                                    Bxfe�  �          @��R�   @�G�?�  A�=qB�B��   @?\)@b�\BE�HB���                                    Bxft8  �          @����ff@��H?���A�{B�B���ff@L(�@O\)B-  B�#�                                    Bxf��  �          @�  ����@��?�(�A��B�Ǯ����@C�
@QG�B0
=B��                                    Bxf��  �          @��ÿ���@���?\A��B�aH����@E@UB1z�B�Q�                                    Bxf�*  �          @��þ\)@{�@p�A���B��\)@��@r�\Bd�B�Q�                                    Bxf��  �          @��ÿ=p�@z�H?�=qA�
=B�aH�=p�@'
=@\(�BO{BӞ�                                    Bxf�v  �          @��ÿY��@k�@��B=qB��f�Y��@��@u�Bl�B�.                                    Bxf�  �          @����}p�@tz�@\)A��B�8R�}p�@�@qG�B`�\B�W
                                    Bxf��  �          @��\�xQ�@^{@1G�B�B�.�xQ�?�\@�=qB~p�B�L�                                    Bxf�h  �          @��ÿ˅@5@HQ�B/�B�uÿ˅?��@��HB�B�C�                                     Bxf�  �          @�
=��{?��R@~{B{��Cp���{�E�@��HB��
CMxR                                    Bxf�  �          @��ÿ�z�@7
=@H��B2�B�LͿ�z�?��@��
B��RC�                                    BxfZ  �          @�G��u@��H?�G�A�ffB��\�u@2�\@^�RBM33B�L�                                    Bxf$   �          @��
���@5�@Z�HBB�B�
=���?c�
@�33B���C��                                    Bxf2�  �          @�33���@6ff@[�BAffB��f���?fff@��
B�{Ck�                                    BxfAL  �          @����=q@)��@a�BG�
B�G���=q?.{@��
B�8RC��                                    BxfO�  �          @���p�@E�@L(�B,(�B�G���p�?��H@�Q�B��C�
                                    Bxf^�  �          @��Ϳ��R@Z=q@:�HBG�B��Ϳ��R?У�@�{B�B�33                                    Bxfm>  �          @�{����@fff@/\)B{B������?��@��Br��B�W
                                    Bxf{�  T          @��ÿ�(�@u�@��A�  B�k���(�@�
@s33BS(�B�aH                                    Bxf��  �          @��
����@z�H@{A��BܸR����@�@���B`z�B���                                    Bxf�0  �          @��׿�(�@w�@
=qA�  B�
=��(�@��@n�RBN�
B���                                    Bxf��  �          @�
=���H@xQ�?�ffA��B� ���H@$z�@Z�HB:�B��\                                    Bxf�|  �          @�z��Q�@h��?��HA�ffB�aH�Q�@!G�@@��B#�Cn                                    Bxf�"  �          @�33��R@l��?�{A_33B�����R@.�R@.�RB�HC0�                                    Bxf��  �          @�{��=q@i��?�ffA�=qB�Q��=q@
=@S�
B?�
B���                                    Bxf�n  �          @��
�^�R@s33?�z�AиRBͳ3�^�R@(�@_\)BU�B��                                    Bxf�  �          @��H��{@�33?�G�A��B�  ��{@N�R@K�B1z�B��                                    Bxf��  �          @��
=�Q�@��>�=q@Y��B���=�Q�@x��@=qA�{B�ff                                    Bxf`  �          @�\)>W
=@�(�>L��@&ffB��>W
=@p��@��A���B���                                    Bxf  T          @��>���@c33?(�AQ�B�=q>���@5�@�RB�HB�                                    Bxf+�  �          @�녿�=q@C33@1G�B�B��
��=q?�\)@x��Bn\)C:�                                    Bxf:R  �          @����{@Z=q@'
=B
�RB�z��{?�  @z�HBj�HCz�                                    BxfH�  �          @�G��˅@`��@�A�=qB��Ϳ˅?��H@p��B`G�C)                                    BxfW�  �          @��ÿ�@=p�@5�B�B��
��?�  @y��BqG�C��                                    BxffD  �          @��ÿ�
=@Vff@�RB�HB�k���
=?�G�@qG�Bd�RC�                                    Bxft�  �          @�G���33@l(�?�A��B�33��33@�@\��BI��B��q                                    Bxf��  �          @�z�L��@_\)@2�\B�RBͳ3�L��?�p�@��
B�Q�B�\)                                    Bxf�6  �          @��
��z�@�  ?h��A8Q�B���z�@S33@5BQ�B�ff                                    Bxf��  �          @������R@��?p��AA�BԊ=���R@QG�@7
=B{B�p�                                    Bxf��  �          @��
���H@w
=?�p�A��B��f���H@#�
@W
=B>
=B���                                    Bxf�(  �          @�(���(�@a�?�
=A���B�\��(�@�@XQ�BJz�C c�                                    Bxf��  �          @�33��
@Z�H?�
=A�(�B�=q��
@(�@G
=B7��C:�                                    Bxf�t  �          @��\��  @_\)?��A��B�W
��  @�@G
=B={B�33                                    Bxf�  �          @�녾B�\@��?��A`  B�\)�B�\@I��@:=qB*��B�\                                    Bxf��  T          @�(�>�\)@�{?���A
=B�ff>�\)@Fff@C�
B2Q�B��q                                    Bxff  �          @�\)��z�@6ff@(�BffB�aH��z�?���@S33B`��C
�                                    Bxf  �          @���r�\?�z�@AG�B33C�=�r�\�aG�@U�B%�C7E                                    Bxf$�  �          @�\)�p  ?B�\@K�B�HC(�{�p  �^�R@I��B33CA�                                    Bxf3X  �          @�\)�u?W
=@@��B��C'���u�5@B�\B��C>}q                                    BxfA�  �          @���[�>�\)@Z=qB3{C/^��[���33@G�B �\CJ=q                                    BxfP�  �          @�z��vff?���@'
=B�C ���vff��\)@<(�B�\C5{                                    Bxf_J  �          @�=q����@
�H?\A�p�C������?�z�@Q�A���C#޸                                    Bxfm�  �          @���o\)?�p�@1G�B
ffCk��o\)�#�
@H��B 33C4�                                    Bxf|�  �          @�z��h��?�G�@5B  CxR�h�ýL��@N{B%�RC4�                                    Bxf�<  �          @�p��c�
@ff@*=qB(�Cp��c�
?��@VffB+��C++�                                    Bxf��  �          @�z��X��?�=q@>{BQ�C���X��>.{@^�RB6�
C1+�                                    Bxf��  �          @����:=q@Z=q�\��G�Cp��:=q@J�H?���A��RC��                                    Bxf�.  �          @�  ��
@g������  B�aH��
@u?�@޸RB�z�                                    Bxf��  �          @����.{@7���33��p�Cp��.{@[���=q�c�
C aH                                    Bxf�z  �          @��\�G
=@<(���=q���C���G
=@N{>�=q@[�C�                                    Bxf�   �          @��H�`  @.�R����V�HC���`  @:=q>���@���CE                                    Bxf��  �          @�p��aG�@@�׿   ��
=C}q�aG�@7�?��\AJ{C                                    Bxf l  �          @�=q�s33@G��^�R�0��C0��s33@��>�{@�(�C��                                    Bxf  �          @�=q�q�@=q�\(��.�\Cu��q�@!�>�
=@�
=C:�                                    Bxf�  4          @��H�vff@  �#�
��\)C���vff?�(�?��Ab=qC�f                                    Bxf,^  �          @�(��XQ�@E�?!G�@��C	���XQ�@��@33A�z�C�                                     Bxf;  �          @�z��9��@g
=�W
=�&ffC � �9��@P��?���A��RC��                                    BxfI�  �          @��H�8Q�@b�\��z��j=qC0��8Q�@N�R?�(�A��C�                                     BxfXP  
�          @���*�H@p��>��@�G�B��3�*�H@E@(�A�C�
                                    Bxff�  �          @�p���@|��<��
>L��B�����@\(�?�Q�A�
=B�#�                                    Bxfu�  �          @�p��-p�@p��>\)?��
B�� �-p�@N{?��HA�  C�                                    Bxf�B  �          @���,(�@a�?���AqB��\�,(�@   @1G�B�C	�                                    Bxf��  �          @���W�@E�0���	C	z��W�@B�\?fffA4Q�C
                                      Bxf��  �          @�(��P  @K��L���\)C���P  @J=q?\(�A*�HC�                                    Bxf�4  �          @��H�I��@Mp��:�H�Q�C���I��@J=q?n{A;�C��                                    Bxf��  
�          @��R�E@`  �������Ck��E@R�\?�ffA�z�C:�                                    Bxf̀  �          @�p��G
=@Y���+��z�Cu��G
=@R�\?�=qAU�C\)                                    Bxf�&  T          @���4z�@c�
�   �ʏ\C ff�4z�@Vff?�ffA�Q�C
                                    Bxf��  
�          @���8Q�@W��s33�B=qC���8Q�@Y��?Q�A&{C:�                                    Bxf�r  �          @����7�@Y���#�
�=qC33�7�@Q�?�{Adz�C@                                     Bxf  "          @�33�c�
@7����
����C��c�
@*=q?���Ac\)C:�                                    Bxf�  �          @�
=�tz�@ff>\)?�=qC+��tz�?�G�?�33At��C33                                    Bxf%d  T          @���o\)?�(�?\A��
C� �o\)?
=q@33A�C+��                                    Bxf4
  �          @���s�
?�\)?���A�  C��s�
?B�\?�(�A�
=C(�=                                    BxfB�  T          @�z��vff?�\?��Aq��CW
�vff?z�H?�{A�
=C%�R                                    BxfQV  �          @�(��~{?�Q�?��Ah(�C )�~{?8Q�?�z�A���C)�=                                    Bxf_�  �          @�p����?�z�?&ffA��C!!H���?fff?�G�A��C'��                                    Bxfn�  
�          @�{��?��?!G�A�RC"�
��?Tz�?�Q�Ay�C(��                                    Bxf}H  
�          @�{�xQ�@33?(�@���C!H�xQ�?��R?��A�z�C                                    Bxf��  T          @�\)�u@p�>�
=@�{C��u?�(�?���A���C�)                                    Bxf��            @���l��@$z�Tz��&�HC@ �l��@(��?
=q@���Cu�                                    Bxf�:  
�          @��
�\(�@8�ÿ���O�
C�3�\(�@A�?�@�
=C
�f                                    Bxf��  4          @��H�A�@G
=��  ����C8R�A�@\��>aG�@0  CG�                                    BxfƆ  �          @���?\)@HQ�Ǯ����C���?\)@_\)>8Q�@�RC�
                                    Bxf�,  
�          @����y��@�Ϳ
=��C���y��@�H?333A	�C8R                                    Bxf��  T          @��H�{�@
=q=��
?xQ�C0��{�?�?���Ai��C޸                                    Bxf�x  �          @�p��y��?���?��@�z�Cn�y��?�
=?�Q�A���C��                                    Bxf  T          @����=q?��ͼ#�
�8Q�C!����=q?�
=?&ffAz�C#�=                                    Bxf�  
�          @�����{?�Q���Ϳ��\C$&f��{?�=q?   @��
C%��                                    Bxfj  
(          @������?�>�ff@���C�H���?���?�ffA��HC!B�                                    Bxf-            @������?�
=�B�\���C�)����?�ff?(��A(�C \                                    Bxf;�  
          @�(���p�?���<��
>�=qC"�3��p�?�?0��A��C%33                                    BxfJ\            @������?�ff?   @�=qC&�)����?(��?s33AAG�C+k�                                    BxfY  
�          @���{?����\)��(�C!�3��{?�=q?
=@���C#\)                                    Bxfg�            @�=q�xQ�@�
����RC=q�xQ�@�\?#�
A=qCs3                                    BxfvN  
�          @�����?�(�>#�
@ ��Ck����?У�?�{Aa��CY�                                    Bxf��  �          @����s�
@�ý�Q쿏\)C��s�
@�?���Ac\)C�                                    Bxf��  �          @����~�R@Q�>L��@   C�=�~�R?�G�?�p�A{�
C(�                                    Bxf�@  
Z          @�Q����?���=�?��
C�R���?�=q?c�
A9�C"@                                     Bxf��  �          @�G�����?����L�Ϳz�C!E����?��
?.{A
{C#L�                                    Bxf��  
(          @�Q��u@33>�{@�z�C{�u?�=q?�Q�A�\)C�=                                    Bxf�2  T          @�
=�H��@A녽�Q쿞�RC
=�H��@+�?�A���C�=                                    Bxf��  
�          @�=q�W
=@4zῙ���tQ�C�R�W
=@B�\>�Q�@��C	�\                                    Bxf�~  
(          @��H�5@]p��n{�<��Ch��5@]p�?fffA5CW
                                    Bxf�$  
�          @��׿�@q녿�z��up�B��
��@w
=?\(�A4��B��H                                    Bxf�  
Z          @��R�J=q@AG�?h��A;�CG��J=q@
�H@�A�=qCu�                                    Bxfp  T          @�Q��g
=?�\)@(�A��C�f�g
=>��@B�\B�HC-}q                                    Bxf&  �          @����{@u�?�p�Ax��B���{@.{@>{B!  C33                                    Bxf4�  
�          @�Q��G
=@L(�?ٙ�A�\)C=q�G
=?�
=@C�
B�RC33                                    BxfCb  	�          @����1G�@`  ?�=qA�Q�C L��1G�@��@G
=B$\)C�\                                    BxfR  �          @�=q�
=q@|(�?�A�33B홚�
=q@%@Y��B4��C��                                    Bxf`�  T          @���0  @c�
?�Q�A��B�p��0  @  @O\)B)G�C�                                     BxfoT  "          @���(��@l��?�ffA���B��(��@(�@K�B%��C	8R                                    Bxf}�  �          @���,(�@p��?�  AqG�B�8R�,(�@)��@<��B��C�                                     Bxf��  	�          @�G��/\)@p  ?�AaG�B�k��/\)@+�@7�BffC��                                    Bxf�F  "          @����,��@l��?���A��RB�Q��,��@#33@>�RB��C��                                    Bxf��  
�          @��
�
�H@�=q?\A�=qB�33�
�H@1�@UB-��C \                                    Bxf��  T          @�33���
@�z�?�\A�(�B�z���
@-p�@eB?�B��                                    Bxf�8  "          @��H�\(�@&ff@�A�G�C޸�\(�?��H@HQ�B"�C ��                                    Bxf��  T          @�����@��?��RA��C�)���?�@��A�p�C$&f                                    Bxf�  
�          @�G���p�@33@g�BH
=C�׿�p�>L��@�G�B�=qC.5�                                    Bxf�*  �          @��H�K�@��@7�BQ�CL��K�>�@eB@ffC+Y�                                    Bxf�  "          @��H�[�@�
@.�RB	{C���[�>�(�@X��B1�
C,�                                    Bxfv  �          @��\�a�@�@&ffBp�C
�a�>��@QG�B*G�C,xR                                    Bxf  
�          @��\�]p�@z�@)��B  C��]p�>�@U�B.=qC,{                                    Bxf-�  T          @�(��s33@{@ ��A�=qC�3�s33?n{@6ffB33C&J=                                    Bxf<h  �          @��\�x��@�?��A�
=C��x��?���@%�BffC"��                                    BxfK  �          @����^{@G�@%�B�C���^{>�@O\)B+\)C,^�                                    BxfY�  
�          @���e�?�33@(�A�(�C��e�>�G�@C�
B!G�C-�                                    BxfhZ  �          @�  �a�?�ff@'�B��C��a�>��@J�HB'z�C/�=                                    Bxfw   �          @��\�]p�@{@#33A�z�CL��]p�?(��@S�
B-(�C)=q                                    Bxf��  �          @�(��\��@(�@=qA홚C�q�\��?k�@S�
B+(�C%!H                                    Bxf�L  
�          @�(��aG�@�\@�RA���C��aG�?@  @R�\B)�
C'�                                    Bxf��  
Z          @��
�b�\@#33@
�HAӮCJ=�b�\?���@I��B!(�C"W
                                    Bxf��  �          @���Z�H@$z�@G�A߮C
�Z�H?���@P  B(z�C"33                                    Bxf�>  "          @�33�R�\@p�@!�A�G�C&f�R�\?aG�@Z�HB4�RC%{                                    Bxf��  �          @�(��U@%@�HA�(�C5��U?��@X��B0=qC"��                                    Bxf݊  "          @�z��]p�@ ��@
=A�33C�]p�?�  @S33B)�HC#�{                                    Bxf�0  
�          @�p��`  @{@=qA�\)Cٚ�`  ?p��@Tz�B*  C%
=                                    Bxf��  	          @�(��O\)@,(�@p�A�Q�CO\�O\)?�{@^{B5�\C!#�                                    Bxf	|  T          @�z��G�@6ff@(�A�(�C	���G�?�G�@b�\B9�
C
=                                    Bxf"  T          @�(��`  @%�@p�A׮C���`  ?���@L��B$�C"
=                                    Bxf&�  �          @���aG�@ ��@z�A�  C�=�aG�?��\@P��B&�RC#�H                                    Bxf5n  T          @�p��^{@"�\@��A�\Cٚ�^{?�G�@UB*��C#��                                    BxfD  �          @�(��c�
@(�@G�A�(�C�H�c�
?z�H@L(�B#
=C$�)                                    BxfR�  "          @�p��g
=@�@ffA���C���g
=?aG�@N{B#�RC&J=                                    Bxfa`  �          @�{�j�H@@�AᙚC���j�H?\(�@L(�B ��C&�                                    Bxfp  T          @���fff@&ff@�A��HC!H�fff?�(�@FffB�HC!G�                                    Bxf~�  �          @���e@4z�?�A�G�C��e?��@>{B{C��                                    Bxf�R  �          @��H�mp�@ ��?�A�  C���mp�?��\@4z�B�C!�                                    Bxf��  T          @��\�qG�@p�?޸RA���C��qG�?�G�@.�RB	�HC!z�                                    Bxf��  �          @��H�i��@�R?�
=A��RCǮ�i��?�Q�@:=qB�\C!�                                    Bxf�D  T          @�(��j=q@{@�\A�  C�j=q?���@?\)B
=C"��                                    Bxf��  "          @�p��tz�@#33?ٙ�A�ffCJ=�tz�?�{@/\)B=qC \)                                    Bxf֐  �          @�p��z=q?�@(Q�B  C \�z=q���
@>�RBQ�C5#�                                    Bxf�6  �          @�Q��|(�@��?�p�A��C�\�|(�?�=q@9��B��C$��                                    Bxf��  �          @���|(�@
=?���A��C  �|(�?�=q@7
=B=qC$��                                    Bxf�  �          @�
=�\)@G�?�z�A��RCW
�\)?��\@1�BffC%�                                    Bxf(  �          @�\)��Q�@{?�
=A��RC���Q�?u@1�B��C&��                                    Bxf�  T          @�\)��z�@   ?�A�G�C(���z�?G�@*=qB=qC)O\                                    Bxf.t  �          @�����z�@
�H?���A��C\)��z�?z�H@*=qA�C&�                                     Bxf=  "          @�Q����?�p�?��A�\)C����?Tz�@"�\A�C(�                                    BxfK�  �          @�
=��@G�?�\A��\C+���?^�R@"�\A��C(+�                                    BxfZf  
�          @�\)���H@�R?޸RA�Ck����H?���@'�A�{C%c�                                    Bxfi  �          @�{����@�
?У�A�  C@ ����?���@#�
A�(�C#��                                    Bxfw�  "          @������@z�?���A
=C�R����?���@33A�Q�C!^�                                    Bxf�X  �          @�p���p�?�?�\)AS\)Cs3��p�?��
?��A��RC&ٚ                                    Bxf��  
�          @�
=���H@�R?uA2=qC����H?��H?�Q�A�(�C!k�                                    Bxf��  T          @�z���  @�?O\)A(�C�R��  ?�=q?�A���C��                                    Bxf�J  �          @�(����@�?��AJ�\C����?�=q@Q�AУ�C��                                    Bxf��  �          @����|(�@�R?�  A���Cp��|(�?�Q�@�HA�\)C#0�                                    Bxfϖ  �          @�����@!G�?��A��C�����?�@%A�{C �R                                    Bxf�<  �          @����~�R@"�\?��HAf{CxR�~�R?���@�
A��C)                                    Bxf��  T          @���У�@vff�B�\�$(�B��У�@mp�?�G�A�B�z�                                    Bxf��  T          @�>���@l���z�����B���>���@��=L��?!G�B���                                    Bxf
.  T          @�=q?�G�@l���
=����B��=?�G�@�z�L���$z�B�\                                    Bxf�  "          @���?�33@h��������B�{?�33@�=q�W
=�)��B�.                                    Bxf'z  �          @��\?�(�@U�(�� �Bm��?�(�@�������RB��
                                    Bxf6   T          @��@�\@G
=�'����Bb�@�\@�  �:�H��
B{��                                    BxfD�  �          @��
@p�@L(���R��G�BQff@p�@w���33����BfQ�                                    BxfSl  �          @�33@  @Tz��\)�陚B_\)@  @\)�����l(�BrQ�                                    Bxfb  T          @�ff@z�@C33�����{BS�@z�@qG���G���
=Bi�                                    Bxfp�  �          @�=q@B�\@�
���ՙ�B(�@B�\@=p�����ffB1{                                    Bxf^  
�          @c�
��?�p���Q����B�#׾�?�{?�A˙�B�ff                                    Bxf�  
Z          @P  �C33>#�
?��\A�C0���C33��p�?uA���C:�)                                    Bxf��  T          @`���Tz�>�
=?@  AI��C,���Tz�    ?\(�Ahz�C4�                                    Bxf�P  �          @�(���33���
?��RA��C4E��33�B�\?��
A���C>p�                                    Bxf��  �          @�
=����\)?�  A���C5����O\)?�G�A�=qC>�                                    BxfȜ  
Z          @�����=q��{?˅A�  C8����=q��=q?��HA�ffCB�)                                    Bxf�B  �          @�G����
?�\?�  AY�C,�����
�L��?�\)At(�C4�R                                    Bxf��  T          @�=q�z=q?У�?=p�A��C^��z=q?�ff?���A�\)C$�R                                    Bxf�  �          @����qG�?�=q?
=A ��C{�qG�?�ff?�A�G�C!                                    Bxf4  "          @�(���?�{?!G�A�C%(���?&ff?���Ah��C+.                                    Bxf�  
�          @�{����?��
?(�@�C&u�����?
=?��AYG�C,#�                                    