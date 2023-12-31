CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230114000000_e20230114235959_p20230115020506_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-15T02:05:06.533Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-14T00:00:00.000Z   time_coverage_end         2023-01-14T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxZ�7@  T          @Å?�Q�@�G�@u�B=qB��?�Q�@?\)@�(�Bdz�B�aH                                    BxZ�E�  T          @�?�\)@��\@j�HB�RB���?�\)@6ff@��Ba(�Bqff                                    BxZ�T�  T          @���?У�@�33@|(�B��B��?У�@@��@�Q�Bc��Bv
=                                    BxZ�c2  �          @�z�?�G�@�z�@���B��B��?�G�@AG�@��HBc(�Bo\)                                    BxZ�q�  �          @�?�z�@��
@c�
BQ�B�=q?�z�@hQ�@�=qBQ�B��{                                    BxZـ~  �          @ə�?�
=@�=q@c33B	z�B�aH?�
=@Vff@��RBPG�BpQ�                                    BxZُ$  �          @�33@
=@��
@z�HB�HB�\@
=@QG�@��\BWG�Bd�\                                    BxZٝ�  �          @�p�@Q�@{�@���B$Bk
=@Q�@ff@�33Bb�HB2��                                    BxZ٬p  T          @�Q�@$z�@n�R@u�B!  B]�@$z�@�R@��B\{B${                                    BxZٻ  �          @ȣ�?�
=@�G�@EA�RB�
=?�
=@mp�@�33B>�Bz                                      BxZ�ɼ  �          @���@&ff@��\@P  A�{B�
=@&ff@{�@�33B7G�BbG�                                    BxZ��b  �          @���@0��@���@B�\A��Brz�@0��@fff@�  B3(�BR33                                    BxZ��  �          @��@\)@hQ�@\)A뙚B^=q@\)@%@c�
B2��B8��                                    BxZ���  �          @��@@  ?��R@�Q�BOQ�Aә�@@  ���
@�Q�Ba��C��
                                    BxZ�T  �          @�p�@`��?=p�@���BAffA>=q@`�׿&ff@���BBffC��H                                    BxZ��  T          @�z�@l�Ϳ\(�@XQ�B&�C�q�@l�Ϳ�Q�@9��B{C�*=                                    BxZ�!�  �          @�
=@��@P  B
=C��)@��C33@
=A�=qC���                                    BxZ�0F  �          @�Q�@������@�A���C���@�����?Y��@�{C���                                    BxZ�>�  �          @�
=@��
�s�
@��A�{C�Q�@��
���?@  @ƸRC���                                    BxZ�M�  
�          @�z�@>�R�S33�.�R��Q�C�@>�R����j�H�2�C��f                                    BxZ�\8  �          @�\)@���{�����h  C��{@��~�R�5����
C��=                                    BxZ�j�  �          @�  @u��p�������C��)@u��G��U���C�Ǯ                                    BxZ�y�  �          @��@�z����ÿQ���(�C�R@�z���\)�(���C��f                                    BxZڈ*  "          @�Q�@����z�?�@w
=C�4{@�����ÿ���  C���                                    BxZږ�  
�          @�@�����p����S�
C��=@��������
�r�\C��                                    BxZڥv  �          Az�@��\���\���\���C�u�@��\�Q���Q��'C�W
                                    BxZڴ  �          A��@�������z��ffC���@��W
=�˅�3��C�#�                                    BxZ���  "          A��@�(���������z�C���@�(��Fff��\�O33C�q                                    BxZ��h  T          A�@����{���ffC��@���Z�H��{�Ap�C�n                                    BxZ��  �          A�@�{������  ���C�7
@�{�y�������5�C���                                    BxZ��  �          A
�H@�33��ff���R�C�.@�33�<������D�C�S3                                    BxZ��Z  
�          A(�@�ff��(���  ���\C�j=@�ff�L�����
�*��C�4{                                    BxZ�   
�          A@����e������@C�0�@��ÿ����(��hC�~�                                    BxZ��  �          A�R@�p��p����{�1C���@�p����Ӆ�YC���                                    BxZ�)L  �          A\)@��H���R���R���
C��@��H�S33����'{C�'�                                    BxZ�7�  �          A�@�������G��w�
C�'�@�����\��  ��
=C��                                    BxZ�F�  T          A�
@��H��\)���n=qC��f@��H��{�|(����C���                                    BxZ�U>  �          A33@��
���
��33��(�C��3@��
���<(����C��\                                    BxZ�c�  �          A�
@θR��=q>��?��\C��\@θR���H���
�+�C���                                    BxZ�r�  T          A�H@�Q���=q��\)�6�RC��q@�Q���\)�W
=��{C�Q�                                    BxZہ0  
�          A
=@У������  �F{C�^�@У��q��J�H��z�C��{                                    BxZۏ�  T          A{@Ǯ����=q���
C��@Ǯ�dz��r�\��(�C�                                      BxZ۞|  T          @�
=@�����=q�����D��C���@�����  �S�
�ծC��                                    BxZۭ"  �          @�\@������R?aG�@�ffC���@�����ff�s33��\)C��H                                    BxZۻ�  �          @�@��R���?fff@�ffC�˅@��R��녿c�
����C��=                                    BxZ��n  T          @�@���ٙ�?\(�@�(�C��
@���ָR��z��4z�C�{                                    BxZ��  �          @陚@����
=>��H@uC�H�@���������R�=p�C���                                    BxZ��  �          @��H@�����{>�33@1�C�s3@�����\)��{�L��C��                                    BxZ��`  T          @�\)@��H���>�z�@z�C��f@��H���ÿ����L��C�\                                    BxZ�  �          @�{@a���
=?J=q@�Q�C�N@a����Ϳ����{C�s3                                    BxZ��  T          @�33@B�\��z�p�����
C���@B�\�ƸR�HQ���
=C��                                    BxZ�"R  T          @�G�@L����=q����AG�C�.@L���љ��+����
C�                                    BxZ�0�  T          @���@B�\��p���=q� ��C�� @B�\��{�%��33C�:�                                    BxZ�?�  �          @���@N{���u���
C�
@N{�ָR�#�
���C��3                                    BxZ�ND  
(          @�z�?��
��p�@33A{33C��?��
��ff�Ǯ�<��C��)                                    BxZ�\�  �          @��R@
�H�߮@��A�(�C�� @
�H���=L��>���C�*=                                    BxZ�k�  �          @�?����У�@UA�G�C�O\?�����  ?�ff@�
=C���                                    BxZ�z6  �          @�ff@z��Ϯ@W�A�Q�C���@z���\)?��AffC��=                                    BxZ܈�  T          A?�\)��z�?+�@��C��=?�\)�����[�C���                                    BxZܗ�  
�          @��H@-p���z�@7
=A��C�
@-p���R?\)@�z�C�K�                                    BxZܦ(  �          A@I����\)?�\AI�C���@I����p��(����{C��H                                    BxZܴ�  �          A	p�@hQ���=q?�(�A9�C���@hQ���
=�O\)��{C���                                    BxZ��t  
Z          A��@�����  >k�?ǮC�4{@�����ff��\�]�C��=                                    BxZ��  �          A�@�ff��(�=�Q�?�RC��H@�ff��G��Q��hQ�C���                                    BxZ���  T          A��@�������>�@N�RC�@�����{��G��=�C�z�                                    BxZ��f  
�          A��@�����z���B�\C���@����������ap�C��=                                    BxZ��  �          A\)@�  ��{���
�(�C�u�@�  ��Q���\�{\)C�~�                                    BxZ��  �          A{@�33��33�k���33C��{@�33��Q��*�H��\)C�G�                                    BxZ�X  �          Az�@�=q���H���w�C�t{@�=q��33�w
=����C�=q                                    BxZ�)�  T          A�H@���z��333���C��@������{��HC�
                                    BxZ�8�  T          A�H@�G����
�(����ffC�xR@�G���{������RC�U�                                    BxZ�GJ  �          A�H@�Q����Dz���z�C���@�Q����
��
=�ffC��                                    BxZ�U�  T          A (�@�Q���ff�5��33C���@�Q����R�����
C���                                    BxZ�d�  �          A�\@�ff��33���{C�p�@�ff�����  ��(�C�]q                                    BxZ�s<  T          A=q@��������U���ffC���@����e��
=�(�C�33                                    BxZ݁�  �          @���@8Q���p����w�
C�O\@8Q�������p���\C�                                    BxZݐ�  �          @�G�@P����=q�%���C�33@P�������G��C��=                                    BxZݟ.  �          @�  @Dz���\)�33��ffC�@ @Dz������=q�z�C�=q                                    BxZݭ�  �          @��H@��H��G��R�\��=qC�f@��H��
=���
�\)C��f                                    BxZݼz  T          @�p�@X�������H���C���@X����=q��=q�	��C�E                                    BxZ��   
�          @�\@B�\�Ϯ������C��\@B�\��(����
��C��)                                    BxZ���  
�          @��@j�H���\�HQ��ÅC�R@j�H���������
C�t{                                    BxZ��l  T          @�Q�@����ƸR���
�.{C��H@�����z��p��up�C��R                                    BxZ��  
�          @��
@�\)��G��˅�@Q�C��R@�\)��Q��P  ���C�#�                                    BxZ��  T          @��
@��R������A�C�L�@��R��ff�@�����
C��{                                    BxZ�^  �          @�z�@��R��ff>��R@"�\C�:�@��R��Q쿾�R�A�C���                                    BxZ�#  T          @�@�G����R>�  @z�C�� @�G���  ��ff�L��C�!H                                    BxZ�1�  �          @�(�@��\��{?��\A&�RC�1�@��\���H��{�0��C��                                    BxZ�@P  �          @��H@U��Ǯ����D��C��@U���ff�Z�H��G�C���                                    BxZ�N�  �          @��
@N{���H��Q���{C�n@N{��{�j�H��33C��=                                    BxZ�]�  �          @��@QG���z��
=�uC�� @QG���������C���                                    BxZ�lB  �          @أ�@^{��G����R�M�C��@^{�����I���߮C��\                                    BxZ�z�  �          @�@W���{��Q��HQ�C�S3@W������
=����C�=q                                    BxZމ�  T          @�(�@N�R����>�?��C���@N�R���ÿ��j=qC�%                                    BxZޘ4  T          @���@k���z�?��RA�
=C���@k���G�?�@�  C�                                    BxZަ�  �          @�ff@����
=@,��A�(�C���@����5?�A���C�h�                                    BxZ޵�  T          @�  @����^�R�@  �׮C�T{@����E������=qC��                                    BxZ��&  �          @�=q@����c�
=u?�C�S3@����Y�������C��                                    BxZ���  
�          @�@��5@>{A�ffC��@��g�?���A�G�C���                                    BxZ��r  �          @��@�ff�c�
@#�
A��C��@�ff��p�?��AF�HC�j=                                    BxZ��  �          @�p�@�33����?���A>�\C��@�33���R����
=C�Y�                                    BxZ���  �          @љ�@q���p�?�33A�ffC��=@q�����>���@*�HC���                                    BxZ�d  �          @�=q@\(����?\(�@��C�9�@\(�����W
=��{C�7
                                    BxZ�
  "          @�p�@5����
�Y����z�C���@5����\�&ff���\C�H                                    BxZ�*�  �          @�Q�@���ff@Tz�B
�C��@���@��@!G�A���C���                                    BxZ�9V  �          @�{@q�?��H@��HBX=qA��@q녾��R@���Bb�RC��                                     BxZ�G�  �          @ָR@��\>aG�@�(�B.�@<(�@��\��ff@�  B'�
C�33                                    BxZ�V�  T          @�=q@����  @�RA��C���@����\)?Tz�A��C��H                                    BxZ�eH  �          @�\)?�33@O\)@�
=BW33B|  ?�33?�=q@�
=B�u�B.                                    BxZ�s�  
�          @�=q@*=q@~{@�G�B'�RB`�H@*=q@�R@�G�B^{B,=q                                    BxZ߂�  
�          @�  @Y��@p�@g
=B&z�B=q@Y��?��\@��BD��A�33                                    BxZߑ:  �          @�p�@��
��z�@A�A�=qC�� @��
�.�R@�
A�=qC��                                    BxZߟ�  T          @�(�@�33>�\)@��B*z�@^�R@�33��G�@��
B$�C���                                    BxZ߮�  T          @��@xQ�?�@���BCQ�A�Q�@xQ�>�  @�
=BU(�@mp�                                    BxZ߽,  �          @�p�@@��?\(�@�G�Br�A~�R@@�׿Y��@���Br=qC�!H                                    BxZ���  �          @�G�@�=q>\@�{BC��@��\@�=q��ff@��HB=��C��=                                    BxZ��x  T          @���@��\���@~�RB�HC�xR@��\��Q�@n{BC���                                    BxZ��  �          @�Q�@`�׾u@��
B^
=C�{@`�׿�G�@�=qBK33C���                                    BxZ���  �          @ə�?�{�   @�z�B���C�?�{�fff@�G�BL�C�Q�                                    BxZ�j  T          @�@{����@�(�B}33C���@{�2�\@�Q�BN�\C�B�                                    BxZ�  �          @�{@J=q��(�?n{Aa�C��@J=q�
=q>���@���C�Ф                                    BxZ�#�  �          @�G���  @0  @j�HBT=qB�uþ�  ?��
@��\B�� Bƣ�                                    BxZ�2\  �          @�p�=�G�@k�@�G�Bjp�B��R=�G�?�(�@ۅB�B�                                      BxZ�A  �          @�ff��  @QG�@�z�B  B��3��  ?�33@��B��RB�ff                                    BxZ�O�  �          @�Q�8Q�@AG�@�33B�W
B�  �8Q�?\(�@�B��{B�aH                                    BxZ�^N  �          @�׿Q�?У�@�Q�B��)B�Q�Q녿
=@�p�B�� CX                                      BxZ�l�  
(          @��fff?�33@�=qB�u�C ��fff���@�\B�z�Ce�                                     BxZ�{�  �          @�\)�p��?B�\@�(�B�ffC33�p�׿���@��B�B�Cm                                      BxZ��@  �          @�׾u?�G�@�B�.B�.�u�u@��B��)C�                                    BxZ���  �          @��H��\���
@�=qB��HC4����\�@�  B���Ce�                                    BxZৌ  �          @�p��+���@�\)B���C6�=�+�� ��@��Bu�RCX�)                                    BxZ�2  �          @�\)�aG�?���@ᙚB�\C )�aG��}p�@�=qB���Cd(�                                    BxZ���  �          @��c�
�O\)@�\)B���C^ff�c�
�/\)@�ffB�8RC|{                                    BxZ��~  
�          @�����?�p�@�{B�Q�C �׿������@�=qB��=CN�{                                    BxZ��$  
�          @���ff?xQ�@��B�\C	aH��ff����@�G�B�G�Cb�)                                    BxZ���  T          @�ff� �׾���@ӅB��fC=)� ����
@ƸRBt�C^�=                                    BxZ��p  T          @��#�
>�@ۅB��B�aH�#�
���
@�ffB�W
C�.                                    BxZ�  "          @��@�?\)@��B�p�Al(�@���G�@�p�B�C�Q�                                    BxZ��  T          @��@#33?G�@��B�k�A�ff@#33���
@�ffB�aHC��\                                    BxZ�+b  �          @�ff@   ?�(�@޸RB��BQ�@   �
=@��HB�#�C��)                                    BxZ�:  �          @���@2�\@ ��@�z�Bn=qB(  @2�\?�@�33B���A8(�                                    BxZ�H�  T          @�z�@C33?�ff@�(�B�A�p�@C33�xQ�@�z�B�#�C�*=                                    BxZ�WT  T          @��@H��?޸R@���Bx�\A�  @H�þ�\)@�  B��\C�w
                                    BxZ�e�  "          A z�@u�?���@�BnA�{@u��8Q�@��Bs�
C��3                                    BxZ�t�  T          A z�@Vff@G�@߮Bs  A��H@Vff�u@��B��=C���                                    BxZ�F  �          A{@hQ�@�@�ffBkQ�A�=q@hQ�=L��@��B}��?Tz�                                    BxZ��  "          A=q@j=q@!�@�G�Bc{B
@j=q?   @�\)B{��@��                                    BxZ᠒  	�          A{@w
=@<��@θRBT33B��@w
=?�  @���Bq�
Ah��                                    BxZ�8  �          @��
@   @�  @�
=BSQ�Bg�H@   @�@�33B���B��                                    BxZ��  �          @��?���@Ϯ@�G�BQ�B��?���@�Q�@�{B?z�B�u�                                    BxZ�̄  
�          A{?ٙ�@ٙ�@�  B�HB���?ٙ�@�Q�@ǮBA�B�33                                    BxZ��*  �          @��?#�
@ƸR@�
=B�B�Q�?#�
@��@У�B[
=B�#�                                    BxZ���  �          A ��?n{@�G�@�B��B��?n{@��@ϮBWG�B�Q�                                    BxZ��v  
�          AG�?�  @ʏ\@�=qB��B�W
?�  @���@�z�BV�\B�G�                                    BxZ�  �          A
=@33@�p�@��RB
=B���@33@���@љ�BL�B���                                    BxZ��  
�          A��?��@��H@�ffB=qB��
?��@��R@���BP�\B�G�                                    BxZ�$h  T          Ap�@%�@���@�B"z�B�@%�@���@�=qB[��Bep�                                    BxZ�3  �          A�@p��@���@�=qB7p�BIG�@p��@*�H@��HBc�
Bp�                                    BxZ�A�  �          A��@g�@�(�@�ffB�Be  @g�@�G�@�=qBEp�B@ff                                    BxZ�PZ  �          Aff@z�H@���@�z�B3��BA=q@z�H@&ff@��
B^p�BG�                                    BxZ�_   T          A   @~�R@\��@��B@33B#�\@~�R?ٙ�@ҏ\Bb�RA��\                                    BxZ�m�  �          @�z�@p��@�  @�G�B(�BHz�@p��@7�@�=qBTBff                                    BxZ�|L  �          @��@xQ�@G�@��
BMffA��H@xQ�?#�
@���Bc�\A                                    BxZ��  T          @�=q@Vff?�@���BWA�
=@Vff>���@��
BlG�@׮                                    BxZ♘  T          @�@����(�@�G�BQ�C�@ @���!G�@_\)BQ�C��\                                    BxZ�>  �          @��
@1�@33@���BM33B�\@1�?J=q@�ffBi�RA
=                                    BxZ��  "          @ָR@dz�?˅@�(�BX
=A�Q�@dz�=�\)@��Bf�?�ff                                    BxZ�Ŋ  T          @ٙ�@Tz�@,(�@�Q�BK�RB\)@Tz�?�z�@���Bk�A�z�                                    BxZ��0  �          @陚@L(�@�
@��HB`  B�@L(�?#�
@�  By�A4��                                    BxZ���  �          @��@u�?�(�@���BY�Aٙ�@u�>�=q@�=qBk  @\)                                    BxZ��|  
�          @�
=@��>8Q�@�z�B&��?�33@�����@�Q�B!33C�q                                    BxZ� "  �          @���@���z�@�Q�B �C��@����@�p�Bp�C��f                                    BxZ��  
�          @�R@˅�Ǯ@p��A�\C�C�@˅����@aG�A�G�C�޸                                    BxZ�n  T          @�G�@�Q��G�@p��A�C��@�Q쿷
=@`��A�  C�Ф                                    BxZ�,  �          @�\)@�G�@#�
@�\)B
=AƏ\@�G�?��\@�  B%\)AN�H                                    BxZ�:�  �          @���@��þ�  @�  BQ�C��R@��ÿ�Q�@���B  C���                                    BxZ�I`  �          @�@��þL��@eA�Q�C�1�@��ÿ�33@Z=qA�z�C�^�                                    BxZ�X  
�          @�ff@���(�@XQ�A�G�C�N@����@H��A�z�C��R                                    BxZ�f�  T          @��@��
��{@5A���C��{@��
��=q@��A���C�
=                                    BxZ�uR  "          @��H@�\)>L�Ϳ����?��@�\)>�{���hQ�@%                                    BxZ��  "          @���@��L�Ϳ�\)���C�>�@�=�G������
=q?Q�                                    BxZ㒞  T          @�G�@�z�c�
�n{��(�C��@�z��R�����
=C��q                                    BxZ�D  �          @��@�(��!G����
��z�C���@�(���{��z��	C���                                    BxZ��  
�          @�=q@�33�k���\�R=qC�%@�33>�  ��\�Q�?���                                    BxZ㾐  "          @�Q�@��Ϳp���ff�{�C�aH@��;�Q�����z�C��)                                    BxZ��6  
�          @���@��
��33�	�����RC��3@��
��������\C��                                     BxZ���  �          @�  @�(����׾L�Ϳ�(�C��=@�(�������fffC�                                      BxZ��  
�          @��@�녿�33�����(�C���@�녿�  �O\)��\)C��                                     BxZ��(  �          @���@�=q�{���^�RC��3@�=q���0������C�1�                                    BxZ��  
�          A��@��   ����:=qC���@���
�������C���                                    BxZ�t  �          A
=@�33�
�H�aG���p�C�Ff@�33��\)��z��ffC�L�                                    BxZ�%  
Z          A
=@�\)�	����\)�6{C�7
@�\)���
=�o\)C��=                                    BxZ�3�  
�          A{@�ff������{C��@�ff��  ��33�X��C���                                    BxZ�Bf  "          A=q@�녿�
=��G��+33C��\@�녿��R����V�RC�~�                                    BxZ�Q  �          A=q@�z��Q��\)��  C�� @�zῈ���&ff���C�f                                    BxZ�_�  T          A=q@��H���\�%���(�C�B�@��H���4z���Q�C��R                                    BxZ�nX  �          @�33@�(���\)�o\)���C��@�(��\(����H����C�H�                                    BxZ�|�  "          @�  @�33��G��q���G�C��{@�33�   ��G����HC���                                    BxZ䋤  T          @�=q@�ff�Q��Z=q��
=C���@�ff�����x����C��H                                    BxZ�J  "          @�  @��1G��dz����
C��q@�����������C�C�                                    BxZ��  
(          @�
=@�=q�0  �h������C�޸@�=q��  ���R�33C�Ff                                    BxZ䷖  �          @�(�@�33�J=q�u��Q�C�!H@�33�ff��  ���C�Ф                                    BxZ��<  "          @�@�Q��>{�~{�{C�C�@�Q������\�Q�C��                                    BxZ���  
�          @�Q�@�z��*�H��(��Q�C�J=@�z��  �����)�HC��H                                    BxZ��  T          @��
@�=q�333��  �Q�C���@�=q���������.{C�C�                                    BxZ��.  
�          @���@�����R��p��
=C�*=@��Ϳ�  ��z��0p�C�1�                                    BxZ� �  
�          @�33@�
=��  ��\)�=�C�E@�
=��z�����I�C�S3                                    BxZ�z  T          @�=q@�Q쿾�R�ƸR�Rz�C��)@�Q�=��
��z��[(�?��\                                    BxZ�   �          A ��@�����������G��C���@���>����Ǯ�K�H@�=q                                    BxZ�,�  "          A�
@��ü��
���
�bQ�C��R@���?�33��p��Xz�A�(�                                    BxZ�;l  �          A�@��?�(���{�i{A�Q�@��@7���{�P=qB�R                                    BxZ�J  �          @�@��\)�X����ffC���@������x����RC��                                    BxZ�X�  
�          @�  @�z���{@0��A�(�C���@�z���{?�ffA  C���                                    BxZ�g^  �          @�=q@����p�@�HA���C�˅@���\?z�H@�C��3                                    BxZ�v  �          @���@�p���=q?�  @�33C�4{@�p����
�����p�C�)                                   BxZ優  "          A z�@�\)��  ?Q�@���C��@�\)��  �Q����
C��                                   BxZ�P  �          A{@�����
=>B�\?�\)C�@�����=q��� ��C�Y�                                    BxZ��  T          A\)@�(���  �����  C��@�(���=q�!G���(�C���                                    BxZ尜  
�          A�
@�z���33�333���
C�~�@�z���Q����w
=C�H�                                    BxZ�B  �          A�@�(��أ׾����C�*=@�(���Q����Up�C��3                                    BxZ���  "          A\)@�G����ͿG����RC���@�G��У��Q����
C���                                    BxZ�܎  T          Az�@�{��Q쿊=q��C��R@�{��=q�%��(�C���                                    BxZ��4  �          A  @�����Q��  �~{C�L�@�����=q�fff��p�C�                                    BxZ���  �          Az�@�Q���\)�  �|��C�<)@�Q������^{�Ə\C�q                                    BxZ��  T          A��@�����@�����C��q@����
���\��ffC��
                                    BxZ�&  �          A(�@��\�ə��@  ��G�C�B�@��\����(�� �HC�J=                                    BxZ�%�  �          A�\@\(��ə��p����G�C�P�@\(���  ���
��C���                                    BxZ�4r  
�          A�@_\)����N�R��  C��@_\)��z�����\)C��                                    BxZ�C  �          A Q�@:=q���H�vff��
=C�S3@:=q������ff�#�C�k�                                    BxZ�Q�  
�          A Q�@1G������a����C�~�@1G���G���{�(�C�Ff                                    BxZ�`d  
Z          A�R@�z���33��\��z�C�W
@�z���p��g���G�C�                                    BxZ�o
  T          A ��@�p���(�����(�C���@�p���ff�g���Q�C�J=                                    BxZ�}�  T          AG�@��\����_\)��=qC�n@��\�����\)�  C���                                    BxZ�V  
�          A@��������H�
��C��H@��~�R��(��1=qC�.                                    BxZ��  �          A{@�z���{���R�z�C���@�z��Ǯ�.{����C���                                    BxZ橢  �          AG�@�����\)���jffC�Ǯ@�������\(��ʏ\C�#�                                    BxZ�H  �          A ��@Z=q���
�<������C�� @Z=q��G������
C�>�                                    BxZ���  "          A Q�@��R���
���R�,  C�=q@��R���
�;���=qC�Ff                                    BxZ�Ք  T          A (�@~�R�ָR��
=�B=qC�U�@~�R��p��HQ����RC�j=                                    BxZ��:  �          @�{@����33�A���p�C��@������������C���                                    BxZ���  
�          @�\)@��H����[���{C�U�@��H��Q��������C���                                    BxZ��  T          @���@��
���
�P����p�C���@��
��Q���
=�
\)C���                                    BxZ�,  
�          @�{@������\���(�C�{@����X������=G�C�`                                     BxZ��  "          A z�@�������z����C�1�@����`  �����9��C�Y�                                    BxZ�-x  
�          @�\@u���ff���\)C�` @u��Tz���G��>33C���                                    BxZ�<  T          @�z�@`  ���H����=qC��@`  �r�\��Q��6=qC�\)                                    BxZ�J�  T          @��@�H�Vff����[G�C��=@�H�   ��{�C�4{                                    BxZ�Yj  �          @�ff@�H�N{�����\�\C�~�@�H������H=qC�                                      BxZ�h  
�          @�@n{�����  �ffC�^�@n{�Vff���H�0{C��R                                    BxZ�v�  T          @�  @����ff��p��O�C�XR@����ff�:�H��p�C��=                                    BxZ�\  T          @�z�@j=q���R@���B'  C���@j=q���R@��\A���C��=                                    BxZ�  
(          @�G�@^{�r�\@��BC
=C�AH@^{��ff@�p�B��C���                                    BxZ碨  �          @�
=@R�\���@�p�B(�HC���@R�\����@�=qA�33C�)                                    BxZ�N  �          @���@�Q����@�Q�B��C�^�@�Q�����@G
=A���C�\)                                    BxZ��  
Z          @�ff@���p�@_\)A�=qC��3@���G�@G�A���C�\)                                    BxZ�Κ  
Z          @���@�z����?���A:�HC�` @�z�����>�Q�@,��C��)                                    BxZ��@  "          @�
=@�����?xQ�@�Q�C�~�@����{�8Q쿧�C�Ff                                    BxZ���  T          @���@�����z���i��C���@���������<(�C�"�                                    BxZ���  T          @�G�@�
=���׿����RC�Y�@�
=��{�
=�~�HC�T{                                    BxZ�	2  T          AG�@�=q�_\)�aG����C���@�=q�,(�������C�|)                                    BxZ��  T          @�ff@����  �5���
C��R@���Tz��fff���C�Q�                                    BxZ�&~  T          @��@��\��33�u��p�C�!H@��\��G����R�w�C��                                    BxZ�5$  T          A ��@�ff��=q��z��$  C�33@�ff����!G���=qC�H�                                    BxZ�C�  
�          A�@��
��{���
�QC�H@��
��ff�9����p�C�E                                    BxZ�Rp  
�          A33@��
��?�AR�RC��R@��
���>�ff@J�HC�h�                                    BxZ�a  
�          A�@�z���  <�>W
=C���@�z���(���
=��RC�AH                                    BxZ�o�  "          Ap�@\��\)?=p�@���C���@\���׾�����C��f                                    BxZ�~b  
�          @�\)@�
=��
=�aG���\)C���@�
=�������H�,��C��)                                    BxZ�  	�          A Q�@N{�陚��33��RC��@N{��\����O�
C�=q                                    BxZ蛮  T          Ap�@�{����>L��?�Q�C��R@�{��녿������
C��                                    BxZ�T  
(          A�@��
�ƸR���B�\C���@��
���H��G���C�f                                    BxZ��  
Z          A�R@�Q���G���{�Q�C�p�@�Q������(��&=qC��                                    BxZ�Ǡ  �          A�@~�R��G�@G�A�(�C���@~�R��33?p��@���C�3                                    BxZ��F  "          A (�@B�\���
@(��A�33C��3@B�\��  ?�  A��C�b�                                    BxZ���  T          A�@���أ�?�\)AC�|)@�����<�>B�\C�:�                                    BxZ��  "          AG�@�����{@(�A|��C�˅@����׮?fff@��C�1�                                    BxZ�8  "          A��@��H�׮?�AR�\C��)@��H��
=?�\@e�C�0�                                    BxZ��  T          A ��@\)�ȣ�@Dz�A�G�C�<)@\)��  ?�ffAO33C�O\                                    BxZ��  
Z          A ��@�  �˅@Q�A�z�C��q@�  ��ff?�\)A ��C�0�                                    BxZ�.*  T          AG�@����{?�R@��RC�Ǯ@����{��R��  C���                                    BxZ�<�  
�          A
=@�33��z�>�  ?޸RC�G�@�33��=q�xQ���=qC�o\                                    BxZ�Kv  T          A�H@����?��@�
=C���@����p��!G����C��=                                    BxZ�Z  
(          A��@�=q��(�?���A6ffC��@�=q��=q>�@QG�C���                                    BxZ�h�  T          A ��@�(�����?�AT(�C�˅@�(���z�?@  @��HC�0�                                    BxZ�wh  
�          A Q�@�z���ff@�AmG�C�@�z���\)?n{@�\)C�W
                                    BxZ�  T          @��@�  ��=q@�HA���C���@�  ��p�?�ffA33C��=                                    BxZ锴  "          @�33@�����@&ffA��HC���@�����
?��RA/\)C���                                    BxZ�Z  �          @��
@��
��{@P��A���C�k�@��
���R@A�
=C��                                    BxZ�   T          @�33@��\��ff@�A�(�C�8R@��\���?�p�A-��C�7
                                    BxZ���  
�          @��@�(���=q@Tz�A�=qC�5�@�(����H@�A��\C��3                                    BxZ��L  �          @��
@�G���z�@p��A�z�C���@�G���  @9��A��C��H                                    BxZ���  T          @��H@����Q�@~{A�
=C��{@������@EA��\C��=                                    BxZ��  
�          @��H@����@\)BffC�g�@���C33@`  BQ�C�AH                                    BxZ��>  �          A Q�@�G�?�  @�
=BW�
Aw
=@�G�=�\)@��HB]��?c�
                                    BxZ�	�  
�          @��@vff?\@�ffBi�A��R@vff>���@ۅBr�\@��R                                    BxZ��  �          @�@���=�\)@�p�BF�H?G�@��׿}p�@��HBC�C�o\                                    BxZ�'0  
�          A   @�\)?xQ�@���B�HA�H@�\)=��
@��Bp�?J=q                                    BxZ�5�  T          AG�@��?@  @���BL�A	G�@����p�@��BN=qC��f                                    BxZ�D|  T          A�H@���?�  @�
=B(��A�
=@���?E�@�{B1=q@�                                    BxZ�S"  �          A�H@�p��u@�{BHffC���@�p�����@���BBp�C��q                                    BxZ�a�  
�          A
=@��>��@\B@��@�=q@���(��@��B@
=C���                                    BxZ�pn  
Z          A�\@�Q�?��@�\)BH�Aw�@�Q�>�z�@�(�BO=q@QG�                                    BxZ�  
(          A�R@��
?���@׮B`p�Av�\@��
<��
@�33Be�>�z�                                    BxZꍺ  �          AG�@��@1G�@�33BH
=A�(�@��?У�@׮BY33A��                                    BxZ�`  "          A��@��H@�H@�{BLz�A�ff@��H?��
@�Q�B[(�Az{                                    BxZ�  	�          A��@tz�@=q@�ffBd
=B �H@tz�?�Q�@�Q�Bt��A�Q�                                    BxZ깬  �          A�@��@O\)@��B>Q�Bz�@��@	��@��
BQ��A�{                                    BxZ��R  T          Aff@��
@Z�H@θRBIffB  @��
@�\@�{B_p�A�ff                                    BxZ���  T          A{@�@�(�@�  B.(�B,�@�@G
=@˅BF�\BQ�                                    BxZ��  "          A��@���@���@�p�B"
=Bff@���@C�
@�Q�B8�A��                                    BxZ��D  �          A{@��@c�
@��B1�B�H@��@!�@��
BF{A�(�                                    BxZ��  "          A��@�=q@E�@�BA{B�@�=q@G�@�33BS�\A�z�                                    BxZ��  "          A��@���?��@�z�BU��A��H@���?�@�=qB^�@�(�                                    BxZ� 6  �          AG�@�33@'
=@�ffBO(�A��
@�33?\@�G�B_G�A�=q                                    BxZ�.�  T          A�@z�?��@�{B�z�Ag�
@z�333@�{B�\C��                                    BxZ�=�  T          A33@[�?�Q�@�\)B{��A�  @[�>�@���B�W
@��H                                    BxZ�L(  T          A=q@�\)?�@��BZG�A���@�\)?8Q�@��Bdp�A                                    BxZ�Z�  �          Az�@j=q@@���Bi
=BQ�@j=q?�
=@��Bx��A�33                                    BxZ�it  �          A�H@<��@.�R@���Bu�
B+
=@<��?\@��B��\A�G�                                    BxZ�x  
�          A�@$z�@  @��HB���B$@$z�?�G�@�33B�G�A�(�                                    BxZ��  "          Az�?�(�?}p�@�p�B��A�z�?�(���=q@�\)B�\C��                                    BxZ�f  �          A�
?�\)?8Q�@�ffB��qA�  ?�\)�
=q@��RB��=C��=                                    BxZ�  "          A�\@#�
?z�H@�p�B��)A��
@#�
��=q@�\)B�G�C���                                    BxZ벲  �          A
=@/\)?޸R@�  B�z�B��@/\)?�@�p�B�.A*�R                                    BxZ��X  �          A�
@g
=?���@��Bz  A�G�@g
=�#�
@�33Bp�C���                                    BxZ���  �          A��@n�R?��
@�G�Bn\)Aˮ@n�R?+�@�\)ByQ�A#
=                                    BxZ�ޤ  �          A\)@l��@%@�\BeB  @l��?�p�@�z�Bv��A�                                      BxZ��J  
�          A�@O\)@(Q�@�ffBo�\B�@O\)?�  @��B�#�Aƣ�                                    BxZ���  �          AG�@S�
@���@�B,ffB]{@S�
@x��@��HBH�BF�\                                    BxZ�
�  "          A   @J=q@�{@�{B/�B`
=@J=q@tz�@��HBK\)BIff                                    BxZ�<  �          @��
@�?��@�Q�B���B  @�?�@�p�B��fAE                                    BxZ�'�  
�          A{@Q�?�z�@�ffB�aHB�
@Q�>u@��\B�ff@�{                                    BxZ�6�  T          A?���?��H@�33B�Q�B�?���<�@�{B�#�?s33                                    BxZ�E.  �          @�\?��?��@�G�B�aHB8��?��>���@�p�B�{AI                                    BxZ�S�  "          @�33@�@&ff@ϮBz��BJ�
@�?�\)@ٙ�B�#�B                                      BxZ�bz  �          A ��@g�@�33@E�A���Bq33@g�@��\@~�RA��Bh��                                    BxZ�q   
�          @��@XQ�@��\@��B
p�Bfp�@XQ�@�z�@��HB&�BW�                                    BxZ��  T          @߮@W
=@=p�@��\BBffB%�@W
=@��@��RBV��B��                                    BxZ�l  T          @�{@5�@�H@��B`��B"
=@5�?˅@�33Bs�
A�
=                                    BxZ�  �          @�ff@K�@p��@���B9�BF��@K�@>�R@�(�BQ  B,��                                    BxZ쫸  T          @���@��?ٙ�@��\B{A��@��?���@���B%\)AQ��                                    BxZ�^  
�          @��@��H@�H@�G�B#=qA�p�@��H?��@��HB1p�A���                                    BxZ��  "          @�ff@��?J=q@��\B��A�@��>W
=@���B  @�                                    BxZ�ת  �          @���@�\)��\)@�
A�ffC��\@�\)�Ǯ@G�A�  C�5�                                    BxZ��P  
�          @�{@�p�@�\@'
=A���A��@�p�?�z�@9��A�=qA�
=                                    BxZ���  
�          @ۅ@��>�(�@g�B�H@�p�@�녽�\)@h��B�HC���                                    BxZ��  �          @�Q�@�\)����@�\)B.33C�Z�@�\)�z�H@���B*\)C��\                                    BxZ�B  T          @�
=@��H���@�BQ�C��\@��H�Q�@�B{C��                                    BxZ� �  
�          @陚@�z��.{@mp�A���C�  @�z��L��@S33A��C�8R                                    BxZ�/�  
�          @�p�@�  �c33@P��A��C��
@�  �}p�@0  A��
C�}q                                    BxZ�>4  T          @�z�@���l��@&ffA��C�H�@������@�A�(�C�0�                                    BxZ�L�  
�          @�@�p��r�\@0  A��C�!H@�p����
@{A��\C���                                    BxZ�[�  
�          @�@�ff���H@G�A�(�C�\)@�ff���?�A^ffC��f                                    BxZ�j&  �          @�{@�z���zᾮ{�*=qC�J=@�z������s33��ffC���                                    BxZ�x�  
�          @�\)@�\)��ff���H�7�C�xR@�\)���R�
=q���HC�)                                    BxZ�r  �          @��
@�z����H����I��C�:�@�z���\)���  C�z�                                    BxZ�  �          @�p�@�����
�B�\��C�U�@����G��z�H��(�C���                                    BxZ���  
Y          @�(�@��
��ff>\)?��C��=@��
��p��!G����C��q                                    BxZ��d  �          @�@�z����H>#�
?�G�C�=q@�z���녿z����C�N                                    BxZ��
  �          @�Q�@�����
?E�@�z�C�޸@����>8Q�?��C���                                    BxZ�а  �          @��@Ǯ��=q?z�@��C�o\@Ǯ���=u?   C�N                                    BxZ��V  "          @�\@�{������d��C�xR@�{��
=��H��C�K�                                    BxZ���  
�          @�G�@�����p���z��M�C�T{@�������ff��\)C��R                                    BxZ���  
�          @�\)@�{��ff��
=�+\)C�!H@�{��
=�
=q��(�C���                                    BxZ�H  �          @��R@�����H�z���(�C��@����
=��ff�z�C�P�                                    BxZ��  �          @�z�@�G��E�@z�A�Q�C�e@�G��U?�Ak
=C�xR                                    BxZ�(�  
�          @�\)@�  �)��@'
=A�(�C�G�@�  �<��@��A��RC�33                                    BxZ�7:  �          @��R@�33�U�?��@���C�\@�33�[�?!G�@�ffC��R                                    BxZ�E�  T          @���@�{�W
=�G�����C�{@�{�O\)��
=�Q�C�z�                                    BxZ�T�  �          @��@�����p������\)C��f@�����Q��33�FffC�@                                     BxZ�c,  "          @�33@��R��ff�P  ��G�C��@��R�����tz���
=C�˅                                    BxZ�q�  
�          @��
@�Q�����/\)��ffC��@�Q���ff�S�
��G�C�q                                    BxZ�x  "          @�z�@������(���
=C���@��������RC��q                                    BxZ�  �          @�Q�@�
=���H?ٙ�AQ�C�|)@�
=��  ?���A
=C�\                                    BxZ��  �          @�  @��R�����(����{C�G�@��R����=q��C���                                    BxZ�j  "          @�G�@�z���  ���H�MG�C��=@�z���Q�����HC�(�                                    BxZ�  �          @��\@������
�����=C��\@������������C�'�                                    BxZ�ɶ  T          @���@������
�����33C���@������R�ٙ��H��C��                                    BxZ��\  
�          @�G�@��
��\)�����C��f@��
��=q��p��NffC�N                                    BxZ��  T          @��H@��H��z�xQ���\C���@��H��  �˅�;�C�R                                    BxZ���  	`          @��\@�(���
=��z��&�RC��{@�(�������\�r=qC�p�                                    BxZ�N  
�          @��H@�\)��  ���c
=C��@�\)��Q��\)��=qC���                                    BxZ��  T          @�@�G��Ǯ��G��8(�C�t{@�G���G��(���=qC���                                    BxZ�!�  T          @���@n�R��\)��R���RC�xR@n�R��{�9����\)C�\                                    BxZ�0@  
�          @��@\(���\)�1�����C��
@\(���z��Z=q��ffC���                                    BxZ�>�  
Z          @�
=@8����ff�'���C�z�@8����z��QG���=qC��                                    BxZ�M�  	�          @��
@[�����"�\��=qC���@[���33�K�����C�.                                    BxZ�\2  
�          @�z�@j=q��\)���R�4��C��q@j=q�����
�H���RC�q                                    BxZ�j�  �          @�@��
��녽#�
��{C�3@��
���ÿ���G�C�*=                                    BxZ�y~  T          @�G�@������
?Tz�@�33C��@�����>�z�@   C�޸                                    BxZ�$  T          @�=q@l����
=@�RA�
=C�xR@l����p�?�\)AQ��C�                                    BxZ��  �          @�
=@��
����?�33A(�C��\@��
���?#�
@�  C��=                                    BxZ�p  �          @�\)@����=q?�z�A.�RC�Z�@����{?h��@�Q�C��                                    BxZ�  �          @�33@����
=?��AffC��@������?�\@|��C���                                    BxZ�¼  �          @�(�@�����z�?�(�A(�C��f@������?0��@���C�B�                                    BxZ��b  T          @�@�G���ff>u?��C��R@�G���ff��\)�	��C���                                    BxZ��  �          @�ff@�z���=q=#�
>��RC��@�z���G���\�~{C��{                                    BxZ��  
�          @�  @Q녿�\)@p  B6C���@Q녿�p�@dz�B+�
C�l�                                    BxZ��T  T          @���8Q�@  @��B�p�B�{�8Q�?�ff@�RB���B�(�                                    BxZ��  "          @�z�>�  ?��@�z�B�\)B�aH>�  ?W
=@�Q�B�B�B�u�                                    BxZ��  �          @�{>#�
?˅@�RB��B��{>#�
?c�
@�\B��B��3                                    BxZ�)F  	�          @�ff?G�@�@��B��B��?G�?��@�\)B��Bs(�                                    BxZ�7�  T          @�G�?�z�@�\@��B�=qBq=q?�z�?�{@�
=B�8RBE�R                                    BxZ�F�  "          @���?(��?�33@�\B�Q�B���?(��?�(�@�\)B�G�Bv�                                    BxZ�U8  �          @�  �k�?�G�@���B�B��f�k�>���@��HB�B��                                    BxZ�c�  
�          @�?У�@ ��@�G�B���BK�
?У�?���@�ffB�=qBQ�                                    BxZ�r�  �          @�@Dz�?�@ۅB�33A��@Dz��@�(�B��HC��                                    BxZ��*  T          @�  @Fff�B�\@߮B��C�)@Fff��33@�z�B~��C�޸                                    BxZ���  
�          @�  @�Ϳ8Q�@�\B�C���@�Ϳ���@�B��\C��                                    BxZ�v  
�          @�  @j�H��@أ�Bv(�C�˅@j�H�#�
@׮BtQ�C�{                                    BxZ�  
Z          @��H@e���Q�@�BzQ�C�  @e��z�H@��
Bv�\C�Y�                                    BxZ��  "          @��\@P  �k�@�33B��RC�  @P  �\(�@ᙚB�\C���                                    BxZ��h  	�          @��@S�
����@�
=B�C�^�@S�
�k�@��B~ffC�AH                                    BxZ��  "          @��H@�ͽu@�z�B��C�C�@�Ϳ0��@�B���C�.                                    BxZ��  
�          @��@\)=��
@��B�=q?�
=@\)���@�z�B�8RC�(�                                    BxZ��Z  T          @�{?u��(�@�G�B���C�  ?u��ff@�\)B��qC�(�                                    BxZ�   
�          @���?�33��\@�
=B�z�C��=?�33�7
=@�  BC��                                    BxZ��  �          @�\)@��Vff@�ffBY(�C��
@��s33@���BHQ�C��                                    BxZ�"L  
(          @�p�@�=q��@�33B
=C��q@�=q���@�p�B	��C�ff                                    BxZ�0�  �          @�@�z��j�H@c�
Aޣ�C�xR@�z��}p�@O\)A�G�C�|)                                    BxZ�?�  "          @�z�@���p�@6ffA�
=C�o\@���z�@{A���C��                                     BxZ�N>  
�          @��@��H��  ?�G�A:=qC��@��H���?���A(�C�Ǯ                                    BxZ�\�  �          @�\)@�ff��z�@G�Av=qC��@�ff��G�?�{AD(�C���                                    BxZ�k�  �          @�ff@�=q��(�?J=q@��C���@�=q��p�>��R@C���                                    BxZ�z0  �          @�(�@����\?�(�A2�RC���@�����?z�H@�p�C�T{                                    BxZ��  
Z          @�\)@q��ə�>�
=@Q�C�xR@q���녽��
��RC�p�                                    BxZ�|  T          @�R@O\)���Ϳ�=q�e�C�j=@O\)�Ǯ�����\C��
                                    BxZ�"  T          @���@G��љ���=q��C�� @G���{�˅�G�
C���                                    BxZ��  
(          @�ff@;���
==���?@  C��=@;��ָR��
=�S33C��\                                    BxZ��n  �          @�
=@tz���Q�?��HA��C���@tz��ʏ\?:�H@�33C��=                                    BxZ��  �          @�p�@����p�?^�R@���C�,�@����
=>�
=@QG�C�\                                    BxZ��  �          @�z�@�z����R?�\@|��C���@�z���\)=��
?��C�z�                                    BxZ��`  �          @�p�@�p����R?��@�G�C���@�p���\)>�?z�HC���                                    BxZ��  �          @�\@q���
=?J=q@��C���@q���Q�>��
@"�\C��\                                    BxZ��  
�          @�ff@XQ���\)�\�B�\C�@ @XQ���{�Tz���
=C�U�                                    BxZ�R  "          @��
@��R�����  �ip�C�b�@��R��33����z�C��                                    BxZ�)�  
�          @�p�@�
=���
>��?��HC�o\@�
=���
����C�q�                                    BxZ�8�  T          @�p�?�
=�U�@��BX\)C�h�?�
=�l(�@���BI��C�C�                                    BxZ�GD  
�          @���?��ÿ��R@�p�B�{C�q?���� ��@�Q�B�C��=                                    BxZ�U�  	�          @��
�����G�@陚B��fCxz����G�@�B�Q�C}��                                    BxZ�d�  �          @�Q�?��
�ٙ�@��B�(�C���?��
�p�@�p�B��C�ff                                    BxZ�s6  �          @���@�\�\��@�Q�Bf=qC�@�\�x��@�Q�BX�HC�8R                                    BxZ��  "          @��@���ff@�ffB@G�C���@����\@��
B1�
C��H                                    BxZ�  T          @�@�H���
@�ffB��C�33@�H��z�@s33A�G�C��H                                    BxZ�(  T          @��\���H��ff@�Q�B��RC^�\���H�L��@��RB��qCn�H                                    BxZ��  "          @�\)�)��@o\)@�  B[Q�B����)��@S33@�\)Bgp�C �                                    BxZ�t  T          A z��Q�@#�
@�BQ�C�f�Q�@@��B��C
��                                    BxZ��  
�          AG����H?L��@���B�z�C�\���H>�z�@���B�z�C+xR                                    BxZ���  	i          A ��?Q�?�z�@��HB�z�Bp(�?Q�?h��@�p�B�� B@33                                   BxZ��f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�@J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�l<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�z�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�9P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�G�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�eB              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�s�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�2V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�O�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�^H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�l�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�+\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�H�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�WN              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�t�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�$b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�3              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�PT  �          @���@����33@Tz�A�C���@����@N{A���C���                                    BxZ�^�  
Z          @��@�z����\@e�A��C��q@�z���p�@^�RA�(�C�XR                                    BxZ�m�  
�          @���@�\)���H@o\)A�p�C�� @�\)��@h��A��HC�U�                                    BxZ�|F  
�          @�ff@���n{@<(�A�
=C��@���r�\@6ffA�\)C���                                    BxZ���  "          @�p�@���l��@;�A��HC�(�@���qG�@6ffA�\)C��                                    BxZ���  "          @�@�ff�tz�@E�A��C�{@�ff�x��@?\)A�C��R                                    BxZ��8  �          @��\@��H����@��RB�C���@��H��p�@�B��C�J=                                    BxZ���  �          @��\@��
�k�@��B�RC��@��
����@�33B�C�p�                                    BxZ�ń  
�          @��\@�
=���@�
=B  C�U�@�
=��33@�{B(�C���                                    BxZ��*  "          @�G�@���>�  @�33B��@33@���>#�
@�33B?�                                    BxZ���  T          @��R@��>��R@�{B)�H@K�@��>W
=@�ffB*{@ff                                    BxZ��v  
�          @��@��
?+�@�  B5��@�(�@��
?\)@�Q�B6Q�@�ff                                    BxZ�   T          @�33@�(�?��@�(�B$  @��@�(�>�G�@�z�B$\)@�=q                                    BxZ��  
�          @�z�@��?(��@��
B+��@�33@��?\)@�(�B,{@�=q                                    BxZ�h  �          @��@�ff��\)@��\B,33C���@�ff�.{@�=qB,�C�#�                                    BxZ�,  �          @��\@���>��@��B1  @���@���>�p�@�p�B1Q�@vff                                    BxZ�:�  �          @�G�@���?(�@�z�B0
=@�G�@���?�\@���B0z�@�Q�                                    BxZ�IZ  �          @�G�@�{?�Q�@�=qB  A4z�@�{?���@��HBA'\)                                    BxZ�X   "          @�\)@���@\)@���B  A�
=@���@
=q@��Bp�A�p�                                    BxZ�f�  T          @��R@���?��H@�G�B.�RA�\)@���?�{@�=qB/�A�{                                    BxZ�uL  T          @�
=@�(�@�R@��B��AŮ@�(�@��@�ffB =qA��                                    BxZ���  
�          @���@E�����@�
=B��{C��@E���  @�ffB��HC��3                                    BxZ���  �          @���@'
=��G�@�
=B��\C�g�@'
=����@�ffB��
C�J=                                    BxZ��>  �          @��@����#�
@�(�B`�C��{@�����G�@�(�B`
=C�C�                                    BxZ���  
�          @�{@J=q���@�p�B�aHC�|)@J=q����@���B�C��)                                    BxZ���  �          @��
@x�ÿ!G�@�Q�Bo
=C�c�@x�ÿ=p�@�  BnffC���                                    BxZ��0  
�          @�p�@vff����@���Bsp�C�C�@vff�L��@���BsG�C�|)                                    BxZ���  
�          @�@w�=#�
@ڏ\Bq��?�@w���\)@ڏ\BqC���                                    BxZ��|  �          @�
=@H���5�@׮Bg�C���@H���;�@�ffBep�C��H                                    BxZ��"  �          A ��@r�\���H@ۅBk(�C���@r�\��@ڏ\BiC�5�                                    BxZ��  
�          A�@k�>�G�@�33By�R@ۅ@k�>�{@�Bz
=@��\                                    BxZ�n  �          @��@�z�?�@׮Bh��@�{@�z�>�(�@�  BiQ�@�                                    BxZ�%  T          @�\)@~{@?\)@���BGz�B�
@~{@:=q@�{BIQ�B�H                                    BxZ�3�  T          @�  @|(�@�(�@��RB)z�B9G�@|(�@��@�Q�B+��B7\)                                    BxZ�B`  �          @���@�\)@.{@�
=B4G�A��@�\)@(��@�Q�B5�RA�                                    BxZ�Q  T          @��H@�33@33@��B1�RA��@�33@�R@�Q�B2�HA��R                                    BxZ�_�  
�          @��
@�(�?�
=@���B7z�A��R@�(�?�{@�p�B8z�A�33                                    BxZ�nR  �          @�p�@�p�?�  @�  B0{A�\)@�p�?�Q�@���B0�A�=q                                    BxZ�|�  T          @�
=@��?���@���B0G�A1�@��?��@�G�B0��A'�                                    BxZ���  
Z          @�@��>�33@�G�B1=q@aG�@��>�z�@�G�B1\)@8Q�                                    BxZ��D  "          @��@��R>��R@�ffB8�@QG�@��R>�  @�ffB8��@&ff                                    BxZ���  
�          @��
@���?��@���BG��@��@���?�@���BH=q@�=q                                    BxZ���  �          @��@�p�>�  @��BU��@C33@�p�>8Q�@��BU@�R                                    BxZ��6  T          @�z�@�Q�>�@�z�Be�?��H@�Q�=u@�z�Be(�?J=q                                    BxZ���  T          @�(�@�  >�\)@��
Be  @n�R@�  >W
=@��
Be�@4z�                                    BxZ��  T          @��
@�\)>\)@��
Be��?��@�\)=�\)@��
Be�R?�                                      BxZ��(  T          @�33@|��>8Q�@���Bo�@'�@|��=�@���Bo33?�                                    BxZ� �  �          @��H@{�>���@أ�Bo�@�@{�>u@أ�Bo=q@`��                                    BxZ�t  T          @��@�G�?   @�
=Bk33@�\@�G�>�G�@�
=Bkp�@�
=                                    BxZ�  T          @�(�@�
=����@�z�Bf�C�W
@�
=�#�
@�z�Bf{C��                                    BxZ�,�  T          @��@���u@��BY  C�� @����@��BX��C�G�                                    BxZ�;f  "          @�33@�Q�   @�z�BZQ�C��\@�Q�\)@�(�BZ�C�y�                                    BxZ�J  �          @�=q@��H=�G�@�=qBW�?�z�@��H=u@�=qBW��?B�\                                    BxZ�X�  �          @���@�  ��G�@ʏ\BZ=qC�Ff@�  �#�
@ʏ\BZ33C��{                                    BxZ�gX  T          @���@��>�Q�@�Q�BW=q@���@��>��R@�Q�BW\)@z�H                                    BxZ�u�  �          @���@�\)>�{@�=qBZz�@�z�@�\)>���@�=qBZ��@q�                                    BxZ���  "          @��H@�\)���
@���B[�C��)@�\)��Q�@���B[�\C��\                                    BxZ��J  
�          @���@��\    @ȣ�BW\)C��)@��\�L��@ȣ�BWQ�C��{                                    BxZ���  �          @�G�@�33��p�@�B_G�C��=@�33��
=@�B_�C�@                                     BxZ���  �          @���@�z�@  @�\)BI��C���@�z�J=q@�
=BIffC�g�                                    BxZ��<  �          @��@�
=��\)@�p�B^G�C�  @�
=���
@�p�B^33C�ٚ                                    BxZ���  
�          @��H@x������@^�RA�33C�� @x�����\@\��A���C�o\                                    BxZ�܈  
�          @��@����1�@�=qB){C�1�@����3�
@���B(\)C�\                                    BxZ��.  "          @�p�@�G�� ��@�p�B-��C��f@�G���@���B-�C��                                     BxZ���  �          @��
@����33@��B>�C���@����
=@�\)B>
=C��q                                    BxZ�z  T          @�p�@�
=���@�BE33C���@�
=��\)@�p�BD��C��                                     BxZ�   
d          @�{@���˅@�p�BPz�C�l�@����\)@��BP
=C�=q                                    BxZ�%�  J          @�  @�z��G�@�Q�B  C�&f@�z���\@\)B��C�                                    BxZ�4l  T          @��H@~�R�(�@��BZG�C��)@~�R�#�
@��BZ{C�h�                                    BxZ�C  T          @�@��J=q@�{B>=qC�1�@��Q�@�B>{C��                                    BxZ�Q�  �          @�  @�\)�u@�33BA�C�� @�\)��Q�@�33BA�C�y�                                    BxZ�`^  "          @�  @�p��8Q�@�\)BD�C��)@�p��@  @�
=BDC�w
                                    BxZ�o  �          @�Q�@�(����
@���BB�C���@�(���ff@�z�BBz�C���                                    BxZ�}�  
�          @���@�\)���@��\B>��C�˅@�\)���@�=qB>�RC��=                                    BxZ��P  
�          @陚@�ff�\)@�p�B5��C�]q@�ff� ��@��B5(�C�C�                                    BxZ���  T          @�Q�@����ff@�  B2Q�C�j=@�����@��B1��C�P�                                    BxZ���  �          @�Q�@��H�#33@���B733C�y�@��H�$z�@�z�B6��C�aH                                    BxZ��B  �          @��@��\���@���B=�\C�,�@��\�{@�Q�B=33C�3                                    BxZ���  �          @��@�\)�	��@��
B7C��=@�\)�
�H@��B7p�C���                                    BxZ�Վ  T          @��@����b�\@��RB��C��@����c�
@�ffB33C��                                    BxZ��4  
Z          @�\@����~�R@\)A�z�C�� @�����  @~�RA��C��{                                    BxZ���  T          @�\@�(���{@n{A�\)C��)@�(���ff@mp�A�ffC���                                    BxZ��  
�          @���@�����G�@���B
(�C��q@�������@�G�B	�RC���                                    BxZ�&  T          @��@����|(�@��HB=qC�3@����}p�@��\B�
C�f                                    BxZ��  T          @�z�@�  ����@�(�B�RC�Y�@�  ���@��
BG�C�O\                                    BxZ�-r  
Z          @��
@���  @�\)B
=C���@���Q�@�
=B��C���                                    BxZ�<  
�          @��@�{��@�z�B�C���@�{��ff@�(�B�C���                                    BxZ�J�  T          @�@�  ��Q�@�  B�HC�@�  ����@��Bz�C���                                    BxZ�Yd  �          @��@������@�=qB��C���@����G�@��BG�C��                                    BxZ�h
  ^          @�ff@�{��
=@�(�BC��@�{��\)@��
BffC�
=                                    BxZ�v�  	�          @��@�G���G�@|��A�C��\@�G�����@|(�A�
=C���                                    BxZ��V  
Z          @��@��
��G�@s�
A���C��
@��
����@s33A�=qC���                                    BxZ���  
�          @�G�@��H��z�@dz�A��C�Ф@��H����@c�
Aڏ\C�˅                                    BxZ���  
�          @�\)@������@K�Aģ�C�N@����G�@K�A�{C�J=                                    BxZ��H  
�          @�@�
=����@
=qA���C���@�
=���@	��A�ffC��                                    BxZ���  �          @��@�����  @uA���C��@�����  @u�A�z�C���                                    BxZ�Δ  "          @�p�@�����Q�@���B�HC��{@�����Q�@�z�B��C��                                    BxZ��:  �          @�\)@�����
@��
B  C��\@�����
@��
B C��=                                    BxZ���  "          @�ff@z�H��=q@���B  C��@z�H��=q@�G�BC��=                                    BxZ���  �          @�ff@�Q���{@�=qB�HC��f@�Q���{@�=qB�C���                                    BxZ�	,  
�          @�p�@z�H���@���B�RC�� @z�H��=q@�z�B�C��q                                    BxZ��  	�          @�@|�����@��B�C�Q�@|�����@��B��C�N                                    BxZ�&x  "          @�z�@]p���Q�@��B'�
C�� @]p���Q�@�33B'��C��)                                    BxZ�5  
�          @�{@y����Q�@��\B�C���@y����Q�@��\Bz�C���                                    BxZ�C�  
�          @�{@�G���p�@�ffB33C��
@�G���p�@�{B{C��{                                    BxZ�Rj  
�          @�@�����
@���B��C�q@�����
@�Q�B�C��                                    BxZ�a  "          @�
=@�{�{�@�\)B�HC�f@�{�{�@�\)BC��                                    BxZ�o�  T          @���@�Q��y��@��HB 
=C��)@�Q��y��@��HA��C��R                                    BxZ�~\  T          @���@���Q�@�=qB�C���@���Q�@��B
=C���                                    BxZ��  �          @���@���+�@�(�B�C�<)@���,(�@��
Bz�C�8R                                    BxZ���  �          @�ff@��
��G�@�G�A�  C��q@��
��G�@���A�C���                                    BxZ��N  �          @�\)@�����@fffA�
=C���@�����@eA��HC���                                    BxZ���  �          @�@��R���R@(�A��C�^�@��R��
=@(�A�\)C�]q                                    BxZ�ǚ  
�          @��R@�����@*�HA��HC�+�@�����@*�HA��RC�*=                                    BxZ��@  
�          @�{@�������@Y��A��C�33@�������@X��A�C�1�                                    BxZ���  
�          @�\)@��R����@7
=A��C��
@��R����@6ffA��C��
                                    BxZ��  
(          @�{@�G�����@a�A�=qC�R@�G�����@a�A�  C�
                                    BxZ�2  
�          @�z�@i����  @�p�Bp�C�l�@i����  @�p�BQ�C�k�                                    BxZ��  �          @�@�������@G�Au��C��q@�������@G�Au�C��)                                    BxZ�~  "          @�(�@����\)@1G�A�33C���@����\)@1G�A�
=C���                                    BxZ�.$  �          @�33@�����R@/\)A�z�C�K�@����
=@/\)A�=qC�J=                                    BxZ�<�  "          @�\@��H��G�@�B��C�o\@��H����@�B�\C�n                                    BxZ�Kp  T          @��@�Q����
@g
=A��C�z�@�Q����
@g
=A�C�y�                                    BxZ�Z  
�          @�@�p���z�@hQ�A���C�  @�p���z�@hQ�A���C��                                    BxZ�h�  
(          @�G�@������@��\B�C�}q@������@�=qB
=C�|)                                    BxZ�wb  T          @���@�{��(�@x��A�{C��H@�{��(�@x��A��C��                                     BxZ��  T          @�@�=q���H@`  A��
C�Q�@�=q���H@_\)A�C�P�                                    BxZ���  T          @�@�33�Y��@�G�Bp�C�H@�33�Y��@�G�B\)C���                                    BxZ��T  
�          @�G�@������@�z�BB
=C��\@������@�z�BB  C��                                    BxZ���  
�          @���@�\)��  @�B8�HC��@�\)��  @�B8�
C��H                                    BxZ���  {          @�R@�G���33@��
B;{C�f@�G���33@��
B;{C��                                    BxZ��F  �          @�ff@�=q�\@��BE  C��q@�=q�\@��BD��C��R                                    BxZ���  �          @�  @��R���\@��\BCQ�C�y�@��R���\@��\BCG�C�s3                                    BxZ��  "          @�Q�@�=q>.{@��RBGff@�@�=q>.{@��RBGff?�p�                                    BxZ��8  
Z          @�@�p���G�@�B6Q�C�"�@�p����\@�B6G�C�q                                    BxZ�	�  
�          @�(�@�
=��@�Q�BD33C���@�
=�z�@�Q�BD(�C��3                                    BxZ��  �          @�@����J=q@�ffB@z�C��{@����J=q@�ffB@p�C���                                    BxZ�'*  �          @�33@�(����H@�G�B<33C�L�@�(����H@�G�B<(�C�E                                    BxZ�5�  
�          @�\@�ff��  @��B1�RC���@�ff��  @��B1��C��{                                    BxZ�Dv  I          @�G�@��
��Q�@��B"�
C�
@��
��Q�@��B"C��                                    BxZ�S  "          @�@��
����@�(�BY��C�h�@��
����@�(�BY��C�]q                                    BxZ�a�  T          @�Q�@�=q����@�33BZ�C���@�=q����@�33BZ��C��=                                    BxZ�ph  
�          @�G�@��z�@�(�BM=qC�y�@��
=@�(�BM33C�n                                    BxZ�  
�          @�  @��׾�@���BIQ�C��@��׾��H@���BIG�C�\                                    BxZ���  
�          @�Q�@����@��BJ��C�:�@�����@��BJ�\C�.                                    BxZ��Z  T          @�
=@�G���  @�
=BH{C�z�@�G����@�
=BH
=C�n                                    BxZ��   �          @�p�@�\)<#�
@�{BH��>��@�\)    @�{BH��=L��                                    BxZ���  �          @�(�@��\>W
=@�G�BCp�@��@��\>L��@�G�BCz�@ff                                    BxZ��L  
Z          @�\@��@w�@g�A���B��@��@w
=@hQ�A�p�B��                                    BxZ���  T          @�@��@��@A�A�  B2�@��@��@B�\Aď\B2                                      BxZ��  �          @�@��@��\@\��A߮B�R@��@��\@]p�A�=qB�\                                    BxZ��>  
�          @��
@�{@�H@��
B�RA�{@�{@=q@�(�B�A�p�                                    Bx[ �  �          @��@��H?�p�@��B�A�(�@��H?�(�@��B�A��                                    Bx[ �  T          @�33@�
=?��@���B"�
A�33@�
=?��@���B#  A�ff                                    Bx[  0  �          @�=q@�{?}p�@��B/  A-G�@�{?z�H@��B/{A+\)                                    Bx[ .�  
�          @��H@�G�?(�@���B.
=@�33@�G�?��@���B.{@�
=                                    Bx[ =|  "          @陚@��\�!G�@�B5�C�q�@��\�&ff@�B5p�C�`                                     Bx[ L"  T          @�ff@�{��\)@��B5{C��)@�{����@��B4�C���                                    Bx[ Z�  T          @�p�@��
����@�
=B;��C�@��
��33@�
=B;�\C��R                                    Bx[ in  �          @���@�
=<�@�33BB\)>���@�
=<��
@�33BB\)>W
=                                    Bx[ x  
�          @�(�@�>���@��BCQ�@j�H@�>�\)@��BCff@^�R                                    Bx[ ��  �          @���@��
���
@�ffBF��C��@��
�#�
@�ffBF��C�Ǯ                                    Bx[ �`  �          @�\@��
��33@�=qBO=qC��R@��
��p�@�=qBO(�C���                                    Bx[ �  
�          @�(�@��;L��@�z�BD�\C��@��;aG�@�(�BD�C���                                    Bx[ ��  T          @�@��\����@�p�B;\)C�f@��\��33@�p�B;Q�C��                                    Bx[ �R  
�          @�{@�  ����@�{B.�C��@�  ��(�@�{B-�HC��3                                    Bx[ ��  T          @�@��þ���@��\B@Q�C�4{@��þ��
@��\B@G�C��                                    Bx[ ޞ  
�          @�ff@�ff�
=q@���B7C��=@�ff�\)@���B7��C���                                    Bx[ �D  
�          @��
@������@�p�B:�
C�AH@�����
@�p�B:C�!H                                    Bx[ ��  "          @��@��
���@�
=B3�C�g�@��
��z�@��RB2�HC�H�                                    Bx[
�  
�          @�(�@��H��@��B�C��@��H�
=@���B��C��q                                    Bx[6  
�          @�Q�@�p����H@�=qB"�HC�b�@�p����R@��B"��C�Ff                                    Bx['�  T          @�=q@�ff�p�@���B�C���@�ff��R@�Q�B�C��                                     Bx[6�  
�          @��@�Q��(��@�33B��C�
@�Q��*�H@��\Bp�C���                                    Bx[E(  
Z          @�G�@�  �O\)@��HB.C�^�@�  �W
=@��\B.�\C�<)                                    Bx[S�  T          @�{@��H��@�(�B�C�\@��H����@��
B�C��3                                    Bx[bt  
�          @陚@�����H@�\)B(�C�ٚ@����p�@�
=B�
C���                                    Bx[q  �          @�Q�@�\)��{@p  A��
C�|)@�\)�У�@p  A�33C�c�                                    Bx[�  �          @��@�\)���@5�A��C��@�\)��z�@4z�A�
=C���                                    Bx[�f  �          @��@�{����@Q�Aڏ\C�h�@�{��(�@QG�A��C�Q�                                    Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[פ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[>.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[L�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[[z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[j               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[x�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[Ъ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[74              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[T�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[c&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[ɰ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[!�  �          @�@����)��?�A/�
C�P�@����+�?�\)A*{C�:�                                    Bx[0:  
�          @�\)@�{���R?�Q�Apz�C�H@�{�G�?�33Al  C��                                     Bx[>�  T          @�@�{�  ?�Q�AP(�C��@�{��?��AK
=C��                                    Bx[M�  "          @�ff@�\)�G�?�G�Az�C���@�\)��\?�(�A33C���                                    Bx[\,  
�          @�p�@�
=��?��A�C���@�
=�33?��\@���C��                                     Bx[j�  
�          @�p�@������?�  A  C�o\@�����H?��HA=qC�Y�                                    Bx[yx  "          @���@��
����?�{Ai��C�3@��
��p�?���Ad��C��\                                    Bx[�  �          @�\@�z���H@ffA��
C��f@�z�� ��@�
A�G�C��R                                    Bx[��  �          @��H@��ÿ��H@(�A��HC��=@��ÿ�G�@	��A���C���                                    Bx[�j  �          @��H@�녿�Q�@	��A�(�C��@�녿޸R@
=A��C��
                                    Bx[�  
�          @�33@��
����?�ffAb�HC�z�@��
���?�G�A]�C�W
                                    Bx[¶  �          @陚@ۅ�G�?��RA<��C��f@ۅ��
?���A7
=C���                                    Bx[�\  "          @陚@�{����?���A*=qC��@�{����?��A%G�C���                                    Bx[�  T          @�p�@�\)�}p�@.�RA���C��@�\)��ff@-p�A��C��\                                    Bx[�  
�          @�\)@��#�
@qG�A�33C��@�����@qG�A��C���                                    Bx[�N  
�          @�  @�Q쾮{@j�HA��C�}q@�Q��(�@j=qA�\C�!H                                    Bx[�  
�          @�Q�@ҏ\�:�H@b�\A��C�Ф@ҏ\�O\)@aG�A��
C�y�                                    Bx[�  
�          @�Q�@θR��@n{A�33C��\@θR�(�@mp�A�=qC�O\                                    Bx[)@  
(          @�@�\)�aG�@n{A�RC�&f@�\)�u@l(�A�33C��f                                    Bx[7�  "          @�  @�\)�z�@��B=qC�U�@�\)�.{@�33B�C���                                    Bx[F�  	�          @�@��
��p�@���Bz�C�0�@��
��@���B{C���                                    Bx[U2  �          @�@����\)@��B{C���@���Ǯ@�33BC�"�                                    Bx[c�  �          @�{@�z�<�@���B�
>�z�@�zὣ�
@���B��C���                                    Bx[r~  
�          @���@��
<#�
@�=qB�H=��
@��
����@�=qB��C���                                    Bx[�$  �          @���@��
=�@~�RA��?�\)@��
<��
@~�RB 
=>W
=                                    Bx[��  �          @�R@�=#�
@��B�>��@��u@��B�C��{                                    Bx[�p  	�          @�Q�@���\)@�G�B�C�\)@����  @�G�B\)C���                                    Bx[�  
�          @���@�(�����@}p�A�z�C�33@�(���\@|��A���C���                                    Bx[��  
(          @�
=@�33����@j=qA�C�!H@�33��@hQ�A�
=C��{                                    Bx[�b  	�          @�Q�@�p���33@��
B{C���@�p���G�@��\B�
C�9�                                    Bx[�  �          @�
=@ʏ\�E�@w
=A��C��@ʏ\�aG�@uA��C�
=                                    Bx[�  
�          @�ff@ȣ׿s33@y��A��RC��3@ȣ׿��@w�A��RC�5�                                    Bx[�T  
�          @�@�=q�Tz�@z�HA��RC�E@�=q�p��@x��A��HC��                                    Bx[�  	�          @�  @�33�Tz�@w�A���C�H�@�33�p��@uA��C��=                                    Bx[�  	�          @��@ҏ\���@e�A�  C��q@ҏ\�&ff@dz�A���C�(�                                    Bx["F  
�          @�@�(���33@w
=A�{C��H@�(����\@tz�A�C�b�                                    Bx[0�  
�          @�@��ÿ�G�@x��A��C�s3@��ÿ�\)@w
=A�\)C���                                    Bx[?�  `          @�Q�@��׿L��@��B��C�9�@��׿n{@�(�B��C��R                                    Bx[N8  
Z          @�  @�
=���@�  Bp�C�"�@�
=�=p�@�
=B��C�y�                                    Bx[\�  	�          @�@�ff��{@�(�BG�C�` @�ff��@��BC���                                    Bx[k�  	�          @�G�@�  �J=q@�(�B�C�h�@�  �k�@��B(�C�Ф                                    Bx[z*  �          @�  @\��p�@��
B�C�G�@\��\@�33B��C���                                    Bx[��  �          @���@�p���Q�@�(�B�HC�>�@�p���\@��
BQ�C���                                    Bx[�v  �          @�@�33�L��@�{B{C���@�33�L��@�B�C��                                    Bx[�  �          @��@��ý�Q�@�\)B=qC��{@��þu@�
=B
=C��q                                    Bx[��  �          @��@���Q�@���B
�RC���@��k�@���B
�C���                                    Bx[�h  �          @���@�=q���
@�BG�C���@�=q�8Q�@�p�B(�C�+�                                    Bx[�  �          @�z�@�Q�
=@_\)A�C�~�@�Q�5@^{A�=qC��)                                    Bx[�  �          @�ff@���ff@K�A�G�C���@���z�@H��A��RC�>�                                    Bx[�Z  �          @�
=@�ff��z�@@  A���C�C�@�ff���\@=p�A�C���                                    Bx[�   �          @��@��
����@%A���C�˅@��
��z�@"�\A��C�o\                                    Bx[�  �          @�
=@�p��fff@*�HA�=qC�p�@�p��}p�@(Q�A�{C�                                    Bx[L  �          @�
=@�33��{@%�A��
C���@�33���H@ ��A��C�=q                                    Bx[)�  �          @���@�\)��Q�@�\A�
=C�o\@�\)��\@{A���C�q                                    Bx[8�  �          @�Q�@�33��=q?�z�AE�C��@�33���?˅A=G�C���                                    Bx[G>  �          @���@�R��{?��@��C��@�R��33?xQ�@�ffC�޸                                    Bx[U�  �          @��H@�{�z�H@33A�ffC�>�@�{����@��A�  C���                                    Bx[d�  �          @�=q@��Ϳ�33@^�RA�C�H�@��Ϳ��
@[�AУ�C��)                                    Bx[s0  T          @�G�@ᙚ�p��@J=qA��C�1�@ᙚ����@G�A��C���                                    Bx[��  �          @���@�=q�n{@Dz�A�z�C�C�@�=q��ff@A�A��C�Ǯ                                    Bx[�|  �          @�ff@�G��:�H@>�RA��HC�\@�G��Y��@<��A���C���                                    Bx[�"  �          @��@��
���@a�A�Q�C��@��
���
@^{A�=qC�xR                                    Bx[��  �          @��@�zΐ33@R�\A�{C��@�zΰ�
@O\)Ạ�C���                                    Bx[�n  �          @�33@�녿��\@%A���C�޸@�녿���@"�\A���C�p�                                    Bx[�  T          @��@���{@2�\A�G�C�o\@���p�@/\)A�  C���                                    Bx[ٺ  �          @�
=@�{���R@A���C�  @�{�˅@�A�p�C���                                    Bx[�`  T          @�  @�p���{@��A�p�C���@�p����H@�A��HC�Ff                                    Bx[�  �          @�G�@�G���p�@�A�ffC�AH@�G���=q@p�A�=qC��H                                    Bx[�  �          @���@陚���@�A|��C���@陚��p�@�\As�C�T{                                    Bx[R  �          @��@�=q��\)@
=qA���C��H@�=q��(�@Ax  C�ff                                    Bx["�  �          @��@�R���@,(�A���C��q@�R��
=@'�A�
=C�ff                                    Bx[1�  
�          @���@�(����@&ffA�z�C���@�(���G�@!G�A���C��                                    Bx[@D  G          @���@�R��Q�@   A�z�C�U�@�R�Ǯ@�A�{C��                                    Bx[N�  �          @���@�\)��z�@p�A��
C�z�@�\)�\@��A���C��                                    Bx[]�  �          @�Q�@�G����R@33A��C�0�@�G����@�RA�{C�Ǯ                                    Bx[l6  �          @�  @�  ��=q@p�A��HC���@�  ��Q�@��A�  C�p�                                    Bx[z�  �          @�Q�@�G����R@�A��\C�>�@�G��˅@ffA{�
C�ٚ                                    Bx[��  a          @�G�@�33��p�@ffAz�RC�S3@�33��=q@�Aq��C���                                    Bx[�(  �          @�  @�녿�z�@Q�A33C���@�녿\@�
Av=qC�%                                    Bx[��  �          @��R@陚���@��A��\C��q@陚����@��A���C�S3                                    Bx[�t  "          @���@��\?��HAip�C�(�@���\)?��A_�C�˅                                    Bx[�  �          @�  @�\��=q@ffA|Q�C��q@�\��
=@�As�C�w
                                    Bx[��  �          @��R@�׿��R@�Az�\C�7
@�׿˅@   Ap��C�Ф                                    Bx[�f  
�          @��@�녿���@��A�C��\@�녿���@��A�  C�/\                                    Bx[�  �          @�Q�@�Q쿱�@�\A�{C�g�@�Q��G�@p�A�33C��                                    Bx[��  �          @�(�@ᙚ���@�A�Q�C���@ᙚ��z�@�RA�(�C�l�                                    Bx[	X  �          @��@�{�z�@�RA��C��{@�{�5@��A�C�.                                    Bx[	�  T          @��
@�Q�8Q�@>{A��C�Ff@�Q쾮{@<��A��\C���                                    Bx[	*�  �          @�@�Q�=���@HQ�A��R?J=q@�Q콣�
@HQ�A��RC���                                    Bx[	9J  T          @�{@�
==�G�@O\)AǮ?k�@�
=��\)@P  A�C���                                    Bx[	G�  �          @�  @�
=>�Q�@W
=A��
@>�R@�
=>.{@XQ�A��H?�
=                                    Bx[	V�  �          @�Q�@޸R>��R@Z�HA�G�@!�@޸R=�G�@[�A�{?p��                                    Bx[	e<  T          @�G�@�\>�  @N�RA�  ?��R@�\=u@O\)Aď\?                                       Bx[	s�  �          @���@��
?(�@q�A���@�G�@��
>Ǯ@s�
A���@Z=q                                    Bx[	��  �          @�@�(�?G�@g
=A�33@�{@�(�?�@i��A��@�z�                                    Bx[	�.  �          @���@�ff?�33@qG�A�  A  @�ff?n{@u�A�(�@���                                    Bx[	��  �          @�\)@Ӆ?��@r�\A���A3�@Ӆ?��@w
=A�Ap�                                    Bx[	�z  �          @��@�p�?��H@�G�A�=qAM�@�p�?�(�@��
B �A+�
                                    Bx[	�   �          @�\)@ȣ�?�(�@�G�A���A�G�@ȣ�?�(�@��BQ�AuG�                                    Bx[	��  T          @��@��@�@~�RA��\A�  @��?�Q�@�33B�A��                                    Bx[	�l  �          @��
@�  ?�G�@�  A�33A{�@�  ?�G�@�33B(�AYG�                                    Bx[	�  �          @���@�G�?�\)@~{A��A=�@�G�?�\)@�G�A���A�                                    Bx[	��  �          @�z�@�z�@�
@{�A�G�A��H@�z�@�
@��B p�A�Q�                                    Bx[
^  �          @�z�@���@�@fffA�G�A��@���?�@n�RA�A��\                                    Bx[
  �          @�33@�  @��@o\)A��A�
=@�  ?�33@w�A��HA���                                    Bx[
#�  "          @�33@��H@�@�Q�A�ffA�\)@��H?�z�@�z�B�HA��                                    Bx[
2P  �          @��H@�{@@n{A��HA��
@�{@ff@w�A��HA�                                    Bx[
@�  �          @�G�@��@-p�@Z�HA�A�z�@��@�R@eA�A��
                                    Bx[
O�  �          @��H@ʏ\@�R@b�\A�z�A�\)@ʏ\?��R@k�A�{A��
                                    Bx[
^B  T          @��H@�=q@�@y��A��A��@�=q@�
@���B�A�{                                    Bx[
l�  �          @�R@�Q�@*=q@�  B{A�  @�Q�@�@�p�Bz�A�                                      Bx[
{�  �          @�@��@�H@�G�B\)A���@��@Q�@�ffB(�A��                                    Bx[
�4  "          @�@���@ff@�{BffA��@���@�\@��HB(�A���                                    Bx[
��  �          @�\)@�@{@���B��A��\@�?�@�ffB33A�33                                    Bx[
��  T          @�@��
@Q�@��\BQ�A��@��
@�@�\)B(�A�(�                                    Bx[
�&  �          @�  @�  ?�@�{BG�A�  @�  ?��
@���BAm�                                    Bx[
��  �          @�  @�{?��@���B�Ah(�@�{?�p�@���B��A;\)                                    Bx[
�r  �          @�@�33?�@���B�HA��@�33?�G�@���BQ�Ag
=                                    Bx[
�  
Z          @�G�@ƸR?�z�@{�A�33Ap  @ƸR?���@���Bz�AHQ�                                    Bx[
�  �          @��@�ff?�@�ffB�RA�G�@�ff?�{@��\BffAs
=                                    Bx[
�d  �          @�  @�z�?޸R@{�A�33A|��@�z�?���@���BAT(�                                    Bx[
  �          @��@��
?���@��
B�HAU�@��
?��@��RB\)A)G�                                    Bx[�  �          @�
=@���?�z�@��B{A�Q�@���?˅@�B��Au                                    Bx[+V  �          @�z�@�\)@6ff@�(�B&  A���@�\)@�R@��\B.\)Aݮ                                    Bx[9�  �          @�@�z�@-p�@���B,�\A�(�@�z�@z�@�
=B4AԸR                                    Bx[H�  �          @�@�{@'�@�Q�B+�HA�@�{@\)@�{B3�
A�                                      Bx[WH  �          @�@��@7�@�(�B133B
=@��@{@��HB:=qA�                                    Bx[e�  �          @�\@�z�@>�R@�Q�B7�B  @�z�@$z�@�\)BAQ�A�z�                                    Bx[t�  �          @��
@�  @>{@�
=B4B�H@�  @#�
@�{B>p�A�z�                                    Bx[�:  �          @���@��@(Q�@�p�B&��A��H@��@�R@��B.��A�\)                                    Bx[��  �          @�p�@�(�@Q�@�G�B+�\A�  @�(�?�p�@��RB2�A�ff                                    Bx[��  �          @�ff@���@�R@�(�B.�\A�{@���@z�@��B6ffA��                                    Bx[�,  T          @��@�p�@�@�p�B1�A�=q@�p�@ ��@�33B9��A���                                    Bx[��  �          @��@��@Q�@���B5z�A�\)@��?ٙ�@�p�B<\)A���                                    Bx[�x  �          @��@�G�@   @�G�B633A�R@�G�@�
@�
=B>�\AÙ�                                    Bx[�  �          @�R@�ff@p�@��B2�RA���@�ff@G�@�B:��A�=q                                    Bx[��  �          @�ff@��@ff@�G�B5�A�(�@��?�z�@�
=B=\)A�{                                    Bx[�j  �          @�\)@��@  @�\)B&�
A�Q�@��?�=q@���B-�HA��
                                    Bx[  �          @�ff@�{@
=@��B33A�p�@�{?�(�@���B�\A�{                                    Bx[�  �          @�  @��
@(�@��B
=Aî@��
@33@���Bz�A��H                                    Bx[$\  �          @�{@�\)@%@�(�B#��A��
@�\)@
�H@��\B,33A�=q                                    Bx[3  �          @�R@��@*�H@��
B"�HA��@��@\)@��\B+�A��                                    Bx[A�  �          @�
=@�@�@��\B!\)AĸR@�?�Q�@���B(�A��R                                    Bx[PN  �          @�@��\@�@�  B�A��R@��\?��
@�p�BQ�A��                                    Bx[^�  T          @�R@�  @�R@���B33A��R@�  ?�=q@�ffB�A�33                                    Bx[m�  �          @�R@���@{@��B{A�33@���?���@�p�B  A��                                    Bx[|@  �          @�R@��\@33@�p�B\)A�Q�@��\?�\)@��B"A��H                                    Bx[��  �          @�@��@Q�@��\B(�A�(�@��?�(�@���B��A��                                    Bx[��  �          @��@��R@��@�z�BA�  @��R?�=q@��\B��A���                                    Bx[�2  �          @�G�@��R@��@�z�B!�HA�\)@��R?�Q�@��\B)�
A�p�                                    Bx[��  �          @��@�
=@��@�p�B"z�Aģ�@�
=?�
=@��
B*p�A�Q�                                    Bx[�~  �          @�33@�33@
=@�33B��A�{@�33?�33@�G�B&\)A��\                                    Bx[�$  �          @�@��\@�@��RB"�HA�@��\?��H@�z�B*�A�ff                                    Bx[��  T          @�\@��@
=@��B)�A��
@��?�\)@��B2�A�z�                                    Bx[�p  �          @�=q@��@=q@�{B-��A�33@��?�z�@�z�B6��A�z�                                    Bx[   �          @�@Å@@��
B33A���@Å?�Q�@���B�\Aw�                                    Bx[�  �          @�  @Ӆ?ٙ�@j�HA�{Af�R@Ӆ?��@s�
A��
A8                                      Bx[b  �          @��@Ϯ?޸R@vffA�z�Ap  @Ϯ?�{@\)A���A=�                                    Bx[,  �          @�\)@�G�?���@�G�A�(�A��@�G�?�ff@��RB  A]��                                    Bx[:�  �          @�{@�  @Q�@�\)B(�A��@�  ?�
=@�{B�A���                                    Bx[IT  �          @��
@�(�@<(�@�G�B  A�ff@�(�@p�@���B&p�A���                                    Bx[W�  �          @���@�ff@,��@�(�B33Aң�@�ff@�R@��
Bz�A�{                                    Bx[f�  �          @�z�@���@(�@�33B  A�
=@���@   @�=qB�A��\                                    Bx[uF  �          @��
@��R@Q�@��BQ�A��@��R?ٙ�@�BQ�A~�R                                    Bx[��  T          @��R@�{�.{@{A�=qC�E@�{��
=@(�A�=qC�7
                                    Bx[��  T          @�z�@�p�����@��A���C��R@�p����@A|��C��                                    Bx[�8  T          A Q�@�33����@e�A�=qC���@�33�0��@a�A��HC�5�                                    Bx[��  �          Aff@ٙ�?n{@�33B�\@�Q�@ٙ�>�@�p�B��@���                                    Bx[��  �          A\)@�p�@*�H@�ffA�
=A���@�p�@�R@�ffB�A�
=                                    Bx[�*  �          A33@�
=?�p�@|(�A�p�A?33@�
=?��@�=qA��A
ff                                    Bx[��  �          Aff@޸R?@  @��HA�z�@�p�@޸R>��
@���A�(�@*=q                                    Bx[�v  �          A�
@���?�\)@}p�A�{AP  @���?���@��A�A
=                                    Bx[�  �          AG�@�Q�?���@h��A�\)Aa@�Q�?�
=@s�
A��A2�\                                    Bx[�  �          AQ�@���?�  @Z�HA�(�A��@���?aG�@b�\A�
=@�Q�                                    Bx[h  �          AG�@���?�{@e�A��
A&�\@���?xQ�@mp�A�p�@�R                                    Bx[%  �          Aff@��
?�G�@r�\A�
=A�\@��
?W
=@y��A�  @�Q�                                    Bx[3�  T          A�
@�33?�Q�@�  A�RA1�@�33?�  @�(�A��H@��                                    Bx[BZ  �          A(�@��?��
@vffA�
=@��@��?(�@|(�A�z�@��
                                    Bx[Q   �          A��@�?��
@eA���@�z�@�?!G�@k�A�=q@�ff                                    Bx[_�  T          A��@�>��@p��AиR?���@����@qG�A��C�l�                                    Bx[nL  �          A��@���Q�@xQ�A�(�C��@���@vffA�{C��                                    Bx[|�  �          A	p�@��=�G�@U�A��\?Q�@�����@Tz�A�{C��                                    Bx[��  �          A	G�@��׽���@g�AǙ�C���@��׿   @e�AŮC�&f                                    Bx[�>  �          Az�@�{��
=@k�Ȁ\C�k�@�{�W
=@g
=A�Q�C��                                    Bx[��  �          A�@���5@�Q�A�ffC�C�@����z�@y��A��
C��                                    Bx[��  �          A�R@��H�u@���A�(�C�G�@��H��z�@xQ�A�C���                                    Bx[�0  �          A=q@�G����@\)A�C�� @�G����@uA�(�C��                                    Bx[��  �          A�@���p�@�Q�A���C�'�@���
=@uA�=qC�j=                                    Bx[�|  �          A��@�(��O\)@j�HAҏ\C��@�(���p�@c�
A�G�C�H�                                    Bx[�"  �          A{@��Y��@hQ�A͙�C�� @����\@`  A�{C�/\                                    Bx[ �  �          A=q@�z῜(�@p  Aԣ�C�O\@�z��33@e�A�Q�C��3                                    Bx[n  �          A��@�\�k�@q�A���C�o\@�\��{@i��AУ�C��H                                    Bx[  �          A�@޸R�У�@�G�A��C�h�@޸R�ff@tz�A��
C��)                                    Bx[,�  T          A\)@�  ��@r�\A���C�W
@�  �
=@b�\A̸RC��3                                    Bx[;`  �          A�@�G���{@aG�A�G�C���@�G���\@U�A��C�(�                                    Bx[J  �          A@�ff��@G
=A��C�N@�ff�Tz�@B�\A�\)C��                                    Bx[X�  �          A{@ᙚ��{@��
A��
C��q@ᙚ�@w
=A��C���                                    Bx[gR  �          A�@�����@{�A��C�G�@���@n�RA�
=C��H                                    Bx[u�  �          A  @���ٙ�@�p�A��C��@�����@{�A��
C�.                                    Bx[��  �          A�
@�  ��\@�  A��\C���@�  �#33@~{A�Q�C���                                    Bx[�D  �          A�@��u@n{A��C�5�@���@dz�AΣ�C�u�                                    Bx[��  �          A33@�\��{@~�RA�z�C���@�\����@s�
AݮC��                                     Bx[��  �          A
=@�=q�(�@fffA�
=C��q@�=q���@`  Aʏ\C���                                    Bx[�6  �          A�R@�p����@X��A�{C���@�p��xQ�@R�\A�(�C�Ff                                    Bx[��  �          A  @����@i��A�ffC�޸@���G�@c33A�Q�C�)                                    Bx[܂  �          A33@��k�@n{A�C�b�@����@dz�AΏ\C���                                    Bx[�(  b          A
=@��ÿ0��@j�HA�\)C�K�@��ÿ�33@c33A��C�~�                                    Bx[��  �          A\)@��þ�
=@Q�A�(�C�ff@��ÿW
=@L��A�33C�Ф                                    Bx[t  �          A�@���333@n{A�=qC�K�@����
=@eA���C�|)                                    Bx[  �          Ap�@�z���H@tz�A�Q�C��@�z�}p�@n�RA�ffC�0�                                    Bx[%�  �          AQ�@���R@��A�  C��H@���z�@���A�\C�\)                                    Bx[4f  �          A�@�z��@�Q�A�C��{@�zῌ��@y��A��C���                                    Bx[C  �          A�
@�
=�#�
@r�\A���C�xR@�
=���@j�HA�\)C���                                    Bx[Q�  �          A�@�ff��=q@S33A�p�C�޸@�ff��G�@HQ�A���C�E                                    Bx[`X  �          Az�@�p��L��@c�
A�(�C��=@�p����\@Z�HAîC�#�                                    Bx[n�  �          A��@��ü��
@}p�A�\C��@��ÿ\)@z�HA�{C��\                                    Bx[}�  �          A�R@�\>#�
@eAУ�?��@�\���
@e�A�  C��q                                    Bx[�J  �          A(�@�Q콸Q�@[�Aď\C���@�Q�\)@X��A��
C�޸                                    Bx[��  �          A��@�z�(�@qG�A�C���@�z῏\)@i��A�z�C���                                    Bx[��  �          AG�@���J=q@>{A�p�C��@����Q�@5A�\)C��                                     Bx[�<  �          AQ�@����R@6ffA�\)C��
@����G�@/\)A��RC�K�                                    Bx[��  �          A(�@�{���@>{A��\C�=q@�{�aG�@7�A���C��)                                    Bx[Ո  �          A��@��R��@C�
A��HC�5�@��R�fff@=p�A��C���                                    Bx[�.  �          A��@��H��R@P��A�p�C���@��H����@H��A�=qC��q                                    Bx[��  �          A�@��;W
=@dz�A�C�1�@��Ϳ5@`  A�C�C�                                    Bx[z  �          @�\)@���W
=?�=qA6ffC�G�@���c33?��@�=qC���                                    Bx[   �          @��R@޸R�g
=?��A\)C�J=@޸R�p��?O\)@�(�C�˅                                    Bx[�  �          @��@�\)�dz�?��HA(Q�C�s3@�\)�o\)?n{@�ffC��f                                    Bx[-l  �          A z�@�ff�L��?��HA'�
C��@�ff�XQ�?z�H@ᙚC�o\                                    Bx[<  �          A z�@�
=�H��?˅A6�\C�>�@�
=�U?�\)A (�C��R                                    Bx[J�  �          A ��@����C�
?\A-�C��
@����P  ?��@�G�C��R                                    Bx[Y^  �          @��@陚�8��?��HA((�C�33@陚�Dz�?��\@��HC��
                                    Bx[h  �          @�
=@�=q�1�?��HA(��C��
@�=q�>{?��
@�RC��R                                    Bx[v�  �          @�@���7�?��A{C�9�@���A�?Y��@�{C��\                                    Bx[�P  �          @��@陚�3�
?�33Ap�C�xR@陚�<��?8Q�@�
=C�                                      Bx[��  �          @�p�@��
�-p�?u@�\)C��@��
�4z�?��@~�RC��                                    Bx[��  �          @�{@�z��1G�?^�R@ə�C��
@�z��7�>�ff@N�RC�b�                                    Bx[�B  �          @��@�R�+�?�Q�AFffC�Ф@�R�9��?��\Az�C�
=                                    Bx[��  �          @���@�\)�<(�@{A�z�C��R@�\)�O\)?�  AL��C���                                    Bx[Ύ  �          @���@�ff�@��@{A�z�C�L�@�ff�S�
?޸RAK33C�E                                    Bx[�4  �          @�z�@�ff�@��@	��A}C�Ff@�ff�S�
?�AC�C�G�                                    Bx[��  �          @�(�@ڏ\�R�\@��A{�C�"�@ڏ\�e�?�{A<z�C�+�                                    Bx[��  �          @��\@�
=�$z�@33A���C��@�
=�8��?��A_�C��)                                    Bx[	&  �          @�(�@�{�
=@(�A�ffC���@�{�p�@AuC���                                    Bx[�  �          @���@�z���R@.{A�\)C�1�@�z����@Q�A�ffC��q                                    Bx[&r  �          @�(�@�G��
=@!G�A�C��q@�G��.{@�Az�HC�o\                                    Bx[5  
�          @��H@�\)�,(�@�A�ffC�u�@�\)�AG�?��A^=qC�K�                                    Bx[C�  �          @��@ᙚ��@$z�A��C�f@ᙚ�*=q@�A��C���                                    Bx[Rd  �          @�33@�R�@�RA�z�C��\@�R�=q?��A]��C��H                                    Bx[a
  T          @���@���?޸RAL  C�g�@��!G�?�{A�\C���                                    Bx[o�  �          @��@��H�=q?��HAG�C��@��H�)��?�ffA�
C�\                                    Bx[~V  �          @�{@�R��R?�p�A,  C���@�R�(�?���A   C���                                    Bx[��  �          @�{@�\)��?�  A.�HC�� @�\)���?���A�C�!H                                    Bx[��  �          @�ff@�G����R?��HA)p�C���@�G����?�\)A��C��q                                    Bx[�H  �          AG�@��
��?�p�A(��C�@��
���?���@��C�K�                                    Bx[��  �          A ��@�=q�
=?�33A (�C�\)@�=q�#33?�  @��
C���                                    Bx[ǔ  �          @�
=@�\)�p�?���A
�\C��@�\)�'�?G�@�(�C�XR                                    Bx[�:  T          A (�@�\)�&ff?���A{C�ff@�\)�0��?0��@�{C���                                    Bx[��  T          Ap�@�p��z�?���@��HC��{@�p��{?+�@�Q�C�
                                    Bx[�  T          A z�@�����?�{@�
=C�5�@����\?=p�@�=qC���                                    Bx[,  �          A Q�@��33?��@�{C���@��(�?.{@�(�C��                                    Bx[�  �          A z�@�p��
=?��@�C�Q�@�p��  ?+�@�G�C���                                    Bx[x  �          A (�@�(���?�\)A ��C��@�(���?=p�@���C�~�                                    Bx[.  �          @�\)@�33��?���A�C�7
@�33��?@  @�p�C���                                    Bx[<�  �          @��@�(��
=?��A\)C�C�@�(��G�?E�@�Q�C��3                                    Bx[Kj  �          @�\)@�33�ff?��RA�HC�Ff@�33��?\(�@ƸRC��                                    Bx[Z  �          @��R@�33�ff?���@�{C�Ff@�33���?:�H@�\)C���                                    Bx[h�  �          A (�@�z��
=q?��\@�\C��@�z��33?#�
@��C��                                     Bx[w\  �          A z�@�\�#33?:�H@�\)C���@�\�(��>�\)@ ��C�e                                    Bx[�  �          A   @����)��>�@^{C�N@����,��<��
>\)C�'�                                    Bx[��  T          A ��@�=q�'�?#�
@��HC�q�@�=q�,(�>8Q�?��\C�4{                                    Bx[�N  �          A (�@�G��'
=?.{@��HC�q�@�G��,(�>W
=?�G�C�.                                    Bx[��  �          A   @�p��9��?+�@���C�S3@�p��>{>��?��C��                                    Bx[��  �          A@�\)�?Tz�@���C��{@�\)���>��@9��C�4{                                    Bx[�@  �          A�@�=q�8Q�?�@�  C��
@�=q�;�=L��>�33C�h�                                    Bx[��  "          A��@��J�H>���@4z�C�s3@��L(��.{����C�b�                                    Bx[�  |          A��A
=�J�H���B�\C�l�A
=�Fff�.{��p�C��                                    Bx[�2  �          Ap�Aff�&ff>.{?�ffC�ٚAff�%������=qC��                                    Bx[	�  �          A��A�\��?�G�@�  C��=A�\�{?�@c33C�=q                                    Bx[~  �          A��A��C33���L��C�w
A��>�R�.{���C���                                    Bx['$  �          A=qAQ��S�
������C��=AQ��K��}p�����C�f                                    Bx[5�  T          A�A  �Q녿
=q�Q�C���A  �HQ쿏\)��z�C�*=                                    Bx[Dp  �          A�A
{�\(����`  C�#�A
{�Q녿�Q���=qC���                                    Bx[S  �          Ap�A(��I���333����C��A(��>{��G���  C��                                    Bx[a�  �          A�A��P  �����z�C��fA��@�׿�\)��C�~�                                    Bx[pb  �          A��A	p��Z�H����  C�'�A	p��G
=�����@z�C��                                    Bx[  �          A
=A�\�*=q�\���C���A�\�����R�B�HC��=                                    Bx[��  �          A�
AQ��1녿}p����RC�l�AQ��#33��p��  C��                                    Bx[�T  �          Az�Az��]p�����G�C�=qAz��I�����H�>{C�                                      Bx[��  �          A�A	��`  ��p��(Q�C��A	��G��ff�fffC�                                    Bx[��  �          A
=A��n�R�����5G�C��A��S�
�!G��x  C�P�                                    Bx[�F  �          A33A(���p����;33C��)A(��n�R�*�H��33C��
                                    Bx[��  �          A�A�\�����Q��iG�C��fA�\�hQ��G���=qC��                                    Bx[�  �          A��A�\��G���p��C�&fA�\�{�����`��C�'�                                    Bx[�8  �          A��A���  �0����Q�C�P�A���G����R��C��f                                    Bx[�  T          A�@�p����
>�
=@(Q�C�޸@�p����
����<(�C��                                    Bx[�  �          Az�A��w
=�}p�����C��qA��fff��(��*�\C�]q                                    Bx[ *  �          Az�A\)�j=q�c�
���C�L�A\)�Z�H�������C��q                                    Bx[.�  �          A��A�R�u��Y�����C���A�R�fff�����  C�p�                                    Bx[=v  �          A{A�\�x�ÿ�����C���A�\�b�\�
=�P(�C��3                                    Bx[L  �          A=qA�����
��  ��HC��fA���p  �G��`(�C���                                    Bx[Z�  �          A�\A=q������ff��p�C�qA=q�n{��
�J�RC��                                    Bx[ih  �          A�A�
��G���33�޸RC�J=A�
�n�R���:�HC�"�                                    Bx[x  �          A
=A����ÿ�G���p�C�FfA��p  ����.�RC��                                    Bx[��  �          A�RA��
=�����{C���A�{���\)�733C�j=                                    Bx[�Z  �          A�\A{���k����\C�� A{�z=q��(��)�C�|)                                    Bx[�   �          A�\Aff��(��xQ���z�C��Aff�vff��\�-G�C��\                                    Bx[��  �          A�Az����׿�G����
C�eAz��n�R����.ffC�.                                    Bx[�L  �          A�
A	p��y���p�����RC��fA	p��h�ÿ��H�%�C��f                                    Bx[��  �          Az�A33��ff�������RC�ǮA33�vff����O\)C��                                     Bx[ޘ            Ap�A	p����׿�33��RC�u�A	p��j=q�(��R�HC�y�                                    Bx[�>  T          A��A
�\�{���G���=qC��=A
�\�e���C�
C���                                    Bx[��  �          A=qA�����ÿ��0��C�aHA���dz��'��}�C��=                                    Bx[
�  �          A�HA���tz��=q�f=qC���A���O\)�HQ���
=C��                                     Bx[0  �          A�A(�������N�HC��)A(��hQ��>�R���HC�w
                                    Bx['�  "          A33A�����Ϳ�Q��8Q�C�
=A���j�H�0  ��C�c�                                    Bx[6|  "          A�
A����녿�Q��   C���A���w��"�\�r=qC��
                                    Bx[E"  
�          A\)A  ���G��@��C���A  �j�H�5���\C�Q�                                    Bx[S�  "          A33A  ������\)�0��C��A  ����4z����HC�7
                                    Bx[bn  T          A�\A�H���ÿ�=q�/\)C���A�H��G��2�\��ffC�'�                                    Bx[q  l          AG�A33������C�%A33��\)�%��z{C�Y�                                   Bx[�  
�          AG�A�
��33��{��HC�g�A�
����"�\�u�C��R                                    Bx[�`  T          A��A����\��33�\)C�q�A���(��%��zffC���                                    Bx[�  �          A�A �����ÿ����C��HA ����33� ���u��C�˅                                    Bx[��  �          A�@�p����Ϳ�p��(��C�  @�p���p��.�R��{C�g�                                    Bx[�R  �          Aff@�����33��\)�{C�\)@�����{��H�n�HC�n                                    Bx[��  c          Ap�@�Q����H�xQ���\)C�` @�Q��������Ip�C�:�                                    Bx[מ  �          A\)@��H������H�B�\C��
@��H��Q��33�%��C�,�                                    Bx[�D  
�          A  @�G����H���R��{C�{@�G����R�  �`��C��                                    Bx[��  �          Az�@��R��ff��(����C���@��R��=q�  �a��C��=                                    Bx[�  �          AQ�@�����33�Q���=qC��@�����녿��8z�C��R                                    Bx[6  
Z          A��@������@  ���
C��=@�����R���333C���                                    Bx[ �  T          A��@�G������=q��
C��f@�G���ff�'
=���C�%                                    Bx[/�  
�          Ap�@�z���z�����`  C���@�z������QG���  C�Z�                                    Bx[>(  
�          A�@������z��g�C�q�@�������U����C�.                                    Bx[L�  "          Aff@��������`��C���@�������S33����C�Z�                                    Bx[[t  
�          Ap�@��
���H����l��C�Ǯ@��
��ff�X����z�C���                                    Bx[j  �          Aff@�  ���H�(Q���{C��@�  �y���e���33C��                                    Bx[x�  T          AG�@�  ����!G��|  C��@�  ����g���\)C��\                                    Bx[�f  
Z          A��@���{�9����\)C��{@��{��w��ģ�C�#�                                    Bx[�  
�          Az�@��H��p��3�
���HC��@��H���u���33C�,�                                    Bx[��  �          AQ�@�  ����0�����\C���@�  ��=q�tz�����C���                                    Bx[�X  
�          A�@�ff��
=���H�E��C��@�ff��(��S�
����C�s3                                    Bx[��  T          A��@У���ff�W
=����C�g�@У������������HC��                                    Bx[Ф  �          AG�@�����G��Q�����C�:�@���������  ��
=C���                                    Bx[�J  T          AG�@�(����
�x������C�O\@�(����H����{C��                                    Bx[��  T          A��@����\)�mp���ffC���@�������
=�ffC�H�                                    Bx[��  "          A  @�{���H��Q��ͅC��H@�{������G���C�E                                    Bx[<  	�          A�
@�z����H����޸RC��f@�z���{��z���RC�^�                                    Bx[�  
�          Az�@��
��p���33�홚C���@��
��ff��{�Q�C��{                                    Bx[(�  T          A��@��H��p��o\)��  C���@��H�����������C�K�                                    Bx[7.  T          Ap�@�ff����S�
��(�C�@�ff��ff��Q����C�y�                                    Bx[E�  	�          A\)@��H����P  ���C�0�@��H��Q���
=���
C���                                    Bx[Tz  
�          A�@��
����N{���HC�4{@��
����������C���                                    Bx[c   T          A��@�33�����Q���{C�:�@�33������H��
=C��
                                    Bx[q�  �          AG�@ʏ\��=q�L����Q�C�)@ʏ\����������
C�k�                                    Bx[�l  T          A�@�Q����
�0����G�C��f@�Q���������G�C��
                                    Bx[�  
�          A�R@����Q����U�C��H@����33�Y�����C�<)                                    Bx[��  �          A�H@�����{�z��c�
C��H@�������e��
=C��\                                    Bx[�^  �          A@�ff����(��Xz�C�+�@�ff��z��\(���Q�C��                                    Bx[�  T          AQ�@�p�����=q�qp�C�XR@�p���{�k���G�C�@                                     Bx[ɪ  �          A33@�33���Ϳ��H�+33C���@�33��=q�G
=��z�C��=                                    Bx[�P  "          A�@�ff�����\)�aG�C���@�ff���
�\(����C��=                                    Bx[��  "          A��@����\)�ff�P��C�O\@�����H�N{��Q�C�3                                    Bx[��  T          Az�@����p�����=qC���@����ff�$z����\C�!H                                    Bx[B  
�          AQ�@�R��G����H���C�XR@�R�����+���z�C���                                    Bx[�  
(          A��@�
=���\��{��HC�<)@�
=��G��5��  C���                                    Bx[!�  
�          A��@�{��
=�����33C��H@�{��\)�#33�|��C��
                                    Bx[04  
�          A@�p���=q���\����C�U�@�p����
����l��C���                                    Bx[>�  
Z          Az�AG���  �J=q���
C�nAG��������8Q�C�S3                                    Bx[M�  �          A�HA ����(����a�C��3A �����
��{�!�C�s3                                    Bx[\&  
�          Aff@����Ϳn{���
C�~�@���G���(��H(�C��H                                    Bx[j�  �          A=qA�\�|(�����ۅC��A�\�c�
�G��Lz�C�7
                                    Bx[yr  �          Ap�@��
��33��  ���C��f@��
�z�H���h(�C��                                    Bx[�  
�          Ap�A z���{����s33C�:�A z����ÿ���C���                                    Bx[��  T          A��@��
���R�����(�C�]q@��
������n=qC��H                                    Bx[�d  T          A�@�p���녾Ǯ�\)C�  @�p����ÿ��8��C��\                                    Bx[�
  "          A=q@�\)��(��G����
C���@�\)������lz�C��)                                    Bx[°  T          AG�@�����(��.{����C�k�@�����Q��33�j�HC�AH                                    Bx[�V  �          AG�@�����Ϳ5��Q�C��\@����G��	���\z�C�s3                                    Bx[��  2          A{@�Q�����8Q쿎{C�Ff@�Q������������C���                                    Bx[�  
J          A\)A���XQ�>�33@
=qC�7
A���W
=��\�J=qC�Ff                                    Bx[�H  
�          Az�A�R�k�?#�
@�Q�C�0�A�R�n�R��������C��                                    Bx[�  �          A�A����Q�>L��?�G�C�qA���{��O\)����C�T{                                    Bx[�  �          A�A{�u=���?
=C��\A{�o\)�^�R��{C��R                                    Bx[):  "          AffA�H����?\)@_\)C���A�H��녾��AG�C��\                                    Bx[7�  
�          A��@�ff���R?
=q@\(�C�C�@�ff��ff�+���\)C�O\                                    Bx[F�  T          A�R@������\=�G�?.{C��3@������R�����(�C�H�                                    Bx[U,  
�          Aff@�������?��\@�(�C���@�����  �L�Ϳ��RC�Q�                                    Bx[c�  �          A
=@��R����?�\@Mp�C�1�@��R��  �(����33C�@                                     Bx[rx  �          A��Ap���?��\@�=qC��fAp���G��\)�aG�C�W
                                    Bx[�  "          A�\@��R����>\@��C���@��R���H�c�
���C��                                    Bx[��  
�          A=q@�����z�?s33@�\)C�� @�����\)��=q���HC�b�                                    Bx[�j  
Z          A@�����Q�>��@?\)C�#�@�����\)�333���C�9�                                    Bx[�  
�          A@��H���
>�  ?ǮC��H@��H���ÿs33��  C�H                                    Bx[��  	p          A��@�����R>#�
?��
C�^�@�����\��\)��z�C��{                                    Bx[�\  v          A�H@��
��p��Q��_�C�]q@��
���]p�����C�aH                                    Bx[�  �          A�@�������z�H��33C��f@������=q���C�Ф                                    Bx[�  �          A=q@�  ��=q��Q��,��C�u�@�  ��p��N{��z�C��                                    Bx[�N  �          A�R@�z����ÿ�z��@(�C��q@�z����\�Z�H����C��                                     Bx[�  �          A�\@�=q�����N�R���C��H@�=q���R��{��
=C�]q                                    Bx[�  �          A�
@�����{����o�
C��@�������w
=��C�>�                                    Bx["@  �          A��@��H��{�0  ���
C���@��H��Q���
=���C�`                                     Bx[0�  �          A(�@�=q�����5���=qC��H@�=q��{��33��  C�K�                                    Bx[?�  �          A\)@����ff�G
=��(�C���@����z��������C�q                                    Bx[N2  �          A
=@�����
�R�\��ffC�� @����Q���p��
=C�8R                                    Bx[\�  �          A�@����=q�h����=qC��=@����(����	�C��                                    Bx[k~  �          A�
@��H��z��h������C���@��H��{��ff�	33C���                                    Bx[z$  �          A{@�ff��z��|(���\)C�n@�ff���H��z��  C�B�                                    Bx[��  �          A
=@�����\�R�\���C���@�����ff����
��C��q                                    Bx[�p  �          A��@�z���ff�R�\����C��q@�z���=q��
=���C�U�                                    Bx[�  �          Aff@��\�����XQ���\)C�S3@��\�����p��	�RC��
                                    Bx[��  �          A��@�=q��33��  ��z�C���@�=q���������HC���                                    Bx[�b  �          A\)@��
��
=�w
=�̸RC��q@��
��ff��z��p�C��q                                    Bx[�  �          A�H@����ȣ��tz����HC�l�@�����Q����
��C���                                    Bx[�  �          A\)@����
=�Q���  C��H@�����H����  C�:�                                    Bx[�T  �          Az�@�G���ff�Z=q��C���@�G���G�������C��H                                    Bx[��  �          A=q@�(���Q��#�
��(�C�\@�(���=q��\)�ܣ�C�J=                                    Bx[�  �          Az�@�����  �p��w�C�N@�����=q��ff����C�g�                                    Bx[F  �          A�H@��R������ff��G�C��=@��R���R���R�Q�C���                                    Bx[)�  T          A=q@���33��\)�֣�C�B�@���ff������C��                                    Bx[8�  �          A��@�z���(����H��33C�g�@�z���ff��p��C�                                    Bx[G8  �          A��@����ȣ�������G�C�W
@�����\)�ə��)�C�`                                     Bx[U�  �          A@�p������33���C���@�p���z���Q��(
=C�%                                    Bx[d�  �          A�@�{�ҏ\��ff��C�t{@�{���H���H�"��C�\                                    Bx[s*  �          A��@��������Q���33C�q�@�����R����C��                                     Bx[��  �          Ap�@�p���  �������C��3@�p������p��%\)C�P�                                    Bx[�v  �          AQ�@��R��\)�����=qC�AH@��R��p������*C�Y�                                    Bx[�  �          A33@�{�ȣ���������C�b�@�{��ff�ʏ\�.z�C�u�                                    Bx[��  �          A
=@�����\)���R��33C��f@�����{���
�'\)C��                                    Bx[�h  �          A
=@��H��(����\�G�C��@��H������Q��>  C��                                    Bx[�  �          A�R@�=q����������\C��@�=q��\)��z��)  C�AH                                    Bx[ٴ  �          A�@�G����
��\)��33C���@�G��_\)��p���RC���                                    Bx[�Z  |          Az�@�p��n�R���H����C��@�p�����
=����C�8R                                    Bx[�   �          A
=@�������������
C��=@����S33��
=��C�o\                                    Bx[�  �          AG�@�R��p���ff��ffC��@�R�S33���H��C���                                    Bx[L  �          A@�Q���=q�����ƣ�C�8R@�Q��^�R��\)���C�/\                                    Bx["�  �          A�@�Q������\)��=qC��@�Q����
��(��p�C�W
                                    Bx[1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[@>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[N�   S          Ap�@������������C��@���Z�H���
�z�C�=q                                   Bx[]�  �          A��@��H�����(���\)C�˅@��H�^�R���H�G�C��\                                   Bx[l0  �          A  @�{��{�������C��@�{�\(���Q����C�aH                                   Bx[z�  �          A=q@�{���H��ff�θRC�@�{�\(�������C�7
                                   Bx[�|  �          A=q@�����  �����ٮC�  @����C33��Q����C��q                                    Bx[�"  �          Aff@�R�s33���\��C�� @�R������R�  C���                                    Bx[��  �          A��@�����G���  ��p�C�ٚ@����I������!�C��                                    Bx[�n  �          A�@�\)��{������(�C�XR@�\)�`  ��Q���
C��=                                    Bx[�  �          A��@ᙚ��=q�����33C��{@ᙚ�X����ff�p�C�*=                                    Bx[Һ  �          A�@�{��Q���\)���C��H@�{�G������"�C�{                                    Bx[�`  
{          A�@�{��Q����H��=qC���@�{�>�R��ff�(�C�`                                     Bx[�  "          A  @�\)����������C�Ff@�\)�,�������{C�k�                                    Bx[��  T          A�\@���������33�(�C���@����e��p��-\)C��                                    Bx[R  �          AQ�@�(���
=��  �  C��@�(��mp��Ӆ�1C�U�                                    Bx[�  T          A��@�z���=q��(��z�C��
@�z��aG���ff�1(�C��{                                    Bx[*�  
x          A  @�33���H���\�{C��R@�33��{���
�J(�C�.                                    Bx[9D  
�          A  @s33��{���R���C���@s33��z���G��E�C�k�                                    Bx[G�  
�          A��@�����\��\)���\C��@��������p��%�
C��\                                    Bx[V�  T          A  @�=q���\�p�����C��)@�=q�p  ���H�
C��=                                    Bx[e6  
�          A�H@�33���w
=��  C��3@�33�tz���
=��RC��)                                    Bx[s�            A�@�z�����J=q��\)C�Q�@�z����\��=q��ffC���                                    Bx[��  �          A��@�{��=q��z����
C�s3@�{�w
=�����(�C���                                    Bx[�(  
Z          A�@�p���33�����C�AH@�p��P����ff�Fp�C���                                    Bx[��  "          A��@j�H������H�/��C���@j�H�HQ���33�c  C���                                    Bx[�t  T          A�@�\)���
��z���C��@�\)�fff��  �>(�C�W
                                    Bx[�  �          A@���������#C��f@���N{��\�TG�C�c�                                    Bx[��  
�          A��@mp���Q���\)�+��C���@mp��O\)�����_��C�e                                    Bx[�f  �          A�@g
=��p���{�)z�C���@g
=�Y����G��_  C�T{                                    Bx[�  �          A{@a����R���\�#�C���@a��mp���Q��[ffC���                                    Bx[��  T          A�H@.{��=q���R�'p�C��@.{��Q���  �d��C�3                                    Bx[X  �          A��@$z���{������C�޸@$z������
=�^�RC��                                    Bx[�  T          A(�@�(����������p�C�~�@�(���z�����,�RC��                                    Bx[#�  T          A��@Tz����H��ff��C�^�@Tz���
=��{�H��C��                                    Bx[2J  �          A�@$z��޸R��G����C�!H@$z�������z��N33C�|)                                    Bx[@�  
(          Ap�@#�
�ۅ���
�{C�@ @#�
��p���{�QffC���                                    Bx[O�  �          A�@^�R�ᙚ��(�����C�%@^�R��������1�
C�j=                                    Bx[^<  �          AG�@>�R��\���� =qC�k�@>�R������\)�?�C���                                    Bx[l�  �          A��@~�R�љ�������C���@~�R�������
�;Q�C��=                                    Bx[{�  
�          A��@����ƸR��Q���ffC�@�����
=��p��5{C��\                                    Bx[�.  
�          A{@��
������\)� C��H@��
��������9
=C�
                                    Bx[��  �          A��@�\)��{���G�C���@�\)�����  �>�\C���                                    Bx[�z  �          Az�@�z���Q���
=��ffC�/\@�z��h������+��C�W
                                    Bx[�   
�          A�H@+�������8C��)@+��U���=q�u\)C�ff                                    Bx[��  
�          A�
@�R��{�����*ffC���@�R���������k�\C�h�                                    Bx[�l  T          AQ�@����ff��p��S(�C�C�@���!G��p���C��                                    Bx[�  �          A�@(��������H�[��C�33@(��33��=qC���                                    Bx[�  �          A\)@��������G��,(�C���@����1G����[��C�~�                                    Bx[�^  
�          A�@h����ff����+
=C��H@h���Tz���{�a�C���                                    Bx[   
�          A\)@j=q��ff��Q��!�\C�W
@j=q�i�����Y�C��f                                    Bx[ �  "          A�@n�R��ff�����
C���@n�R�mp��ᙚ�U  C���                                    Bx[ +P  "          A  @HQ�����ff�33C��
@HQ���{��p��Sz�C��R                                    Bx[ 9�  �          Ap�@����z���
=�$�C�E@����\)�����f=qC���                                    Bx[ H�  �          Ap�@!G���G���\)�?�\C�5�@!G��L(����}C�&f                                    Bx[ WB  
�          A��@p���33��  �ap�C�� @p�����p�C��)                                    Bx[ e�  �          Aff@(Q��J=q����yz�C�޸@(Q���33�3C��                                    Bx[ t�  �          A�@#33���
��ff�[z�C�#�@#33���33p�C���                                    Bx[ �4  
�          A��@!G���\)��
=�^�C�` @!G���\��HǮC�l�                                    Bx[ ��  
Z          AG�@N�R��=q��=q�J�C��q@N�R�{����~��C��H                                    Bx[ ��  T          AG�@���(���p��E
=C�]q@��AG�� Q��C���                                    Bx[ �&  T          Aff@33����{�X(�C�{@33������  C�!H                                    Bx[ ��  �          A?�����{��p��Fz�C�33?����E�� ��G�C��)                                    Bx[ �r  
�          Ap�?�����33���
�0�C�Z�?����w���
=�v(�C�<)                                    Bx[ �  "          A{@\)��z����
�K(�C��f@\)�.{�{#�C��R                                    Bx[ �  �          AG�@������=q�6��C���@��XQ������w=qC���                                    Bx[ �d  
�          A�@!G���z���G��-33C��3@!G��l(����H�m��C�'�                                    Bx[!
  �          A{@G
=��\)�ᙚ�UQ�C�%@G
=��ff� z���C���                                    Bx[!�  T          A��@(Q���z��߮�R�C�@(Q�������(�C��                                    Bx[!$V  �          Ap�@?\)���
��33�Y�C���@?\)��� ���fC�p�                                    Bx[!2�  
�          AQ�@u������{�v��C�� @u?p����G��|ffA\Q�                                    Bx[!A�  "          A=q@Q��ۅ�����=qC���@Q���(��ᙚ�Q��C���                                    Bx[!PH  "          A@Q������{��
C��
@Q�������[��C��                                    Bx[!^�  "          A�?�{��
=��33�P��C�E?�{�"�\��aHC�"�                                    Bx[!m�  �          Ap�?�Q���\)���\�(�C�
=?�Q���{��ff�[G�C�>�                                    Bx[!|:  T          A{@K��Å���\�(�C��q@K��������Z�HC�,�                                    Bx[!��  
�          A=q@U���\)��ff�'��C��@U��b�\��R�cC��q                                    Bx[!��  �          A�
@Q���Q���
=�
=C�aH@Q���{��{�`��C��                                    Bx[!�,  �          AQ�?�  �ᙚ��z��z�C��q?�  ���
��(��P��C�%                                    Bx[!��  "          A@   �׮���\���C�AH@   ����R�[�C��                                    Bx[!�x  "          A��@z���������HC��@z��������
�bQ�C�7
                                    Bx[!�  
�          A��?�p��Å��ff�2Q�C��?�p��s�
���\�y�
C��                                     Bx[!��  �          A�@	��������z��3  C�(�@	���a���p��v��C���                                    Bx[!�j  "          A�@   ��G��ə��7G�C��@   �]p����\�{�HC�                                      Bx["   �          A
=@33��=q����,
=C�aH@33�s33��{�o�
C��R                                    Bx["�  "          Ap�@   ��(���{�  C��q@   ����z��M�C�u�                                    Bx["\  �          A�R@���
=��G��{C�\@���33��\�_�
C�P�                                    Bx[",  �          A  @C�
������  �9Q�C�@C�
�@����z��u�
C��3                                    Bx[":�  "          A(�@G���
=��=q�#G�C��=@G��qG����b�\C�˅                                    Bx["IN  �          A  @Dz����H��=q�33C��\@Dz���ff��=q�Z=qC��                                    Bx["W�  T          A��@#�
��{�����(\)C�9�@#�
�y����\)�k�C��                                     Bx["f�  	�          A��@,(���z������C�n@,(����R����`
=C�K�                                    Bx["u@  	�          A@P����\)��\)�33C�Z�@P�������G��T��C�h�                                    Bx["��  �          Aff?z�H��  ����f��C���?z�H� �����z�C��3                                    Bx["��            A{?�����G���\)�t
=C��R?��ÿ�(��
={C�                                      Bx["�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx["��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx["�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx["�$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx["��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx["�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx["�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#%              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#3�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#BT              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#nF              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#�*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$ �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$;Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$J               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$X�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$gL              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[$��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%4`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%C              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%Q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%`R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%Ղ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[%��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&-f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&J�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&YX              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&g�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&v�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&�J              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&Έ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx[&�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['	               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['&l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['5              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['C�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['R^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['a              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx['o�   2          A#�@�  ���
���
�@z�C�!H@�  ���R�
�\�g
=C��                                    Bx['~P  �          A'33@P������
=��C�Ф@P����33��
=�G�RC��{                                    Bx['��  
m          A(��@������H��  ��HC�p�@����Y���
=�Up�C��
                                    Bx['��  
�          A'\)@�������G��"33C�"�@���4z��Q��Q�C��=                                    Bx['�B  �          A'\)@��R���H�׮� z�C�{@��R�,����R�M33C���                                    Bx['��  
(          A$z�@����������
�!(�C���@����:=q���Q�RC�q                                    Bx['ǎ  T          A&�H@ҏ\��ff��p���RC��f@ҏ\�(���\)�?  C���                                    Bx['�4  
�          A(  @У���
=��{��C�@У��Q������D(�C��{                                    Bx['��  
�          A*�H@�=q��G���33� ffC��3@�=q����G��E��C��R                                    Bx['�  T          A,(�@�(��Y����Q��#33C��H@�(��G���  �9
=C��)                                    Bx[(&  "          A.�R@�p��%�����(\)C���@�p�=�����  �5�?:�H                                    Bx[(�  "          A/�@�(��7
=��33�(Q�C��)@�(������z��7�
C�o\                                    Bx[(r  @          A'�
@�33���R���R�B�RC�l�@�33���
�Q��j�C���                                    Bx[(.  �          A ��@����G���z��P��C�]q@����=q����v�HC�g�                                    Bx[(<�  T          A�?�\��p�����:��C���?�\�I�����
��C���                                    Bx[(Kd  �          A�R@{��  �˅�;�C���@{�>{���u�C�`                                     Bx[(Z
  �          A��@Z=q����G��{C��{@Z=q��(���p��X�C�                                    Bx[(h�  
�          A�\@|����������-��C���@|���S33�\)�j�RC��                                    Bx[(wV  
�          Ap�@Vff��
=�����8
=C��q@Vff�6ff�{�vQ�C��3                                    Bx[(��  T          Ap�@)�������ff�U�C�K�@)�����R�
�R.C��                                    Bx[(��  g          A�@h����
=�ȣ��.�\C��{@h���>�R�����k33C�aH                                    Bx[(�H  7          Az�A��?��ÿ��\����A,��A��@�
��{��(�AB�H                                    Bx[(��  T          A   A��?У׿#�
�k�A�A��?�  �#�
�W
=A"ff                                    Bx[(��  
�          A%A!��?��H��
=��=qA�A!��@ �׿��7�A4z�                                    Bx[(�:  T          A$z�Ap�������H�
�\C�FfAp�?5���R���@��
                                    Bx[(��  �          A#33A33������R��C�ٚA33>�z���\)�G�@�                                    Bx[(�  "          A"�RA (����
�����C�� A (�>�����\)�p�@5                                    Bx[(�,  "          A$Q�@�  �C�
�ȣ����C��@�  �5��{�*��C�H�                                    Bx[)	�  
Z          A!p�A=q�-p���
=��=qC�ǮA=q��  ��(�� (�C�z�                                    Bx[)x  �          A�HA   ��z�?�R@q�C�C�A   ��녿�{��Q�C�~�                                    Bx[)'  T          A  A
=���H?z�H@���C�A
=���
�W
=��
=C���                                    Bx[)5�  
�          AG�A ����G��s33��
=C���A �������(Q���  C��=                                    Bx[)Dj  
�          Ap�@�p���(�������=qC��)@�p��|�������)
=C��\                                    Bx[)S  �          A
=@��
��R�&ff�}�C�aH@��
�أ���Q��G�C���                                    Bx[)a�  
�          A'�@�����e��
=C�J=@����\)�ƸR���C�o\                                    Bx[)p\  
F          A,(�@���������G�RC�R@��Ϳ��R����p�C���                                    Bx[)  �          A'�@�z���=q��\�9�\C�K�@�z��Q��33�j�C��                                    Bx[)��  �          A�@�=q��=q�����F��C��@�=q�޸R�z��uz�C��                                    Bx[)�N  "          A��@�ff��(���(��/p�C���@�ff������VffC�{                                    Bx[)��  "          Az�@�
=���R��  �.�
C���@�
=�333���e(�C���                                    Bx[)��  T          A�\@����z���  ��C�H@���E���  �<p�C�e                                    Bx[)�@  T          A&�R@�����z���33�(�C���@����dz����KG�C��                                    Bx[)��  �          A%p�@�=q�У������C�AH@�=q�����ff�@(�C��                                    Bx[)�  T          A$��@�\)�ָR��{�
=C���@�\)������P�RC�S3                                    Bx[)�2  
Z          A"�R@s33�׮����!33C���@s33��=q���b��C��H                                    Bx[*�  "          A"ff@����
=���H��\C��R@���~{��ff�N(�C���                                    Bx[*~  
	          A ��@�\)���������RC���@�\)�@  ����1��C���                                    Bx[* $            A{@�����
=��z���=qC�Y�@����qG��ʏ\�1
=C�9�                                    Bx[*.�  "          Ap�@�����Q��Y������C���@����a���p��C�z�                                    Bx[*=p  
�          A Q�@������H�}p���ffC�Z�@��������7
=��
=C�8R                                    Bx[*L  
�          A�
@e�����(����C�1�@e�������\��HC���                                    Bx[*Z�  �          A	�@�33��
=�p����=qC�^�@�33��{�]p���Q�C���                                    Bx[*ib  
�          A��@���Q������G�C�9�@����H�����p�C���                                    Bx[*x            A{@�=q���\�c�
��C��=@�=q���0�����C�q�                                    Bx[*��  
�          A/\)Aff��Q�@  �|(�C�ФAff��p��&ff�\Q�C�33                                    Bx[*�T  
�          A4��Ap���
=������33C�` Ap��Tz����H�=qC�&f                                    Bx[*��  �          A7�@�������ۅ���C�"�@������33�6z�C�,�                                    Bx[*��  
�          A4(�@�������z����C��\@���33� ���5�RC��                                    Bx[*�F  "          A2ffA��C33�θR��C��A��&ff����C��3                                    Bx[*��  "          A2ffA	p��(���љ��Q�C�s3A	p��k�������C�<)                                    Bx[*ޒ  �          A*ff@�ff��z���33����C��=@�ff�p���=q�ffC�j=                                    Bx[*�8  T          A1AQ���(��~{����C�k�AQ��>�R��=q��C��R                                    Bx[*��  T          A2�HA	������߮��C��HA	�?�z�����=q@�\)                                    Bx[+
�  T          A1A��(����  ��\C��3A�?ٙ���=q��
A/�
                                    Bx[+*  �          A;�A#
=��p���G���ffC�޸A#
=?�\���R��{@8��                                    Bx[+'�  g          A9A$(�>�  ��(���\)?��A$(�@���
=���A7�                                    Bx[+6v  A          A:ffA&�R?c�
���H�υ@���A&�R@%��
=���HA^�H                                    Bx[+E  �          A:�HA#\)?
=������{@S33A#\)@p���Q���=qAX��                                    Bx[+S�  g          A:{A (�?�p���ff��
=AffA (�@U��33��G�A��                                    Bx[+bh  �          A8Q�A?�ff��G����
A)��A@p��������A���                                    Bx[+q  S          A0��A�@�  ��������A��A�@�z��=q�K�A�{                                    Bx[+�  T          A33@�ff�&ff�hQ���C��f@�ff���������&(�C���                                    Bx[+�Z  �          A{?�ff�
=�B�\��33C��f?�ff��������z�C���                                    