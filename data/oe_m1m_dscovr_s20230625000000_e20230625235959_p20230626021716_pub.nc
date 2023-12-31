CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230625000000_e20230625235959_p20230626021716_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-06-26T02:17:16.850Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-06-25T00:00:00.000Z   time_coverage_end         2023-06-25T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx����  
�          Az��>{��33��33�T�
Cn}q�>{�n{�\)�o�Cgh�                                    Bx���f  	�          A(��.{���H����^��CoO\�.{�\(��{�yp�Cg��                                    Bx���  
�          AQ��'����R��j�Cn\�'��A��	�#�Ce�                                    Bx��Ʋ  7          A�
����p���(��=Cvff����!��\)��Cl�R                                    Bx���X  "          AG��P  ��ff��\)�Z�Ciٚ�P  �Tz��Q��r�
Ca�)                                    Bx����  T          Az��  �e��z��{��Cm�)�  ����33��Cb��                                    Bx���  "          A33?G������¤  C�,�?G�>����ªA�z�                                    Bx��J  	�          A\)?�
=��R�\)(�C�(�?�
=?+��\)�A�                                      Bx���  ?          AQ�?�(��ff��Hk�C�{?�(��L���{¢\)C�b�                                    Bx���  7          A��?ٙ������ffu�C���?ٙ��(���G��fC�w
                                    Bx��-<  "          AQ�@*�H��ff��  C���@*�H��������C���                                    Bx��;�  �          A�
�8����(������NG�CoaH�8���vff��Q��h��Ci{                                    Bx��J�  �          A�
���R�i����
=�2\)CU� ���R�.{����BQ�CNz�                                    Bx��Y.  "          Aff��p���{��
=��  CW���p��������\�\)CS#�                                    Bx��g�  q          A��ᙚ��  ��
�h(�CY���ᙚ��(��C33��(�CX
=                                    Bx��vz            A����������?��
@�\)CZT{��������>��H@C33CZ�                                    Bx���   
�          A\)�z=q�����У��(�
Cm��z=q���\��G��B��Chs3                                    Bx����  "          A�H�c33�ʏ\��(��%G�Cp�R�c33�������@�Cl�                                    Bx���l  �          A���%���߮�<Cvp��%��p���\)�Y  CrJ=                                    Bx���  
�          A33�У���
=�����Y�C|�ÿУ���z�����w33Cx��                                    Bx����  �          A33����\)�\)�g33CvT{���e��z��=Co�H                                    Bx���^  
�          A���������[��Ct����tz��G��w
=Cn��                                    Bx���  T          A��Q���(������Z��CvaH�Q��tz����v��Cp��                                    Bx���  
�          A��������\)�cz�Cw����e������Cr0�                                    Bx���P  "          A�R���R������m�C{𤿾�R�Z�H�
=qL�Cv��                                    Bx���  T          A������z���=q�d33C~�R����s33�ff�)Cz�                                    Bx���  q          A�ÿ����ff���H�ZCͿ����(����x33C{p�                                    Bx��&B  7          A{���H��(����R�]\)C~
���H�����G��z�\Cz+�                                    Bx��4�  T          A��޸R��Q���33�`  Cz�)�޸R�z�H�33�|z�Cv�                                    Bx��C�  T          A(������\��\�l�Cy�R���\���33k�Ct5�                                    Bx��R4  T          Aff��\)��G������`��C~�q��\)�}p��ff�}�HCz޸                                    Bx��`�            A\)�\��=q��\)�f�C|�=�\�n�R����\Cwٚ                                    Bx��o�  �          A(��z�H��{���Uz�C��H�z�H����G��s33C���                                    Bx��~&  T          A�H�p������� ���k�\C�j=�p���j�H�	�� C�)                                    Bx����  "          Az�}p���z��(��p��C��R�}p��`������C~B�                                    Bx���r  
�          A�}p�������R�tC��3�}p��X���
=�C}��                                    Bx���  ?          Ap��aG���  �
ffQ�C���aG��5�����C|��                                    Bx����            A�׿fff�j�H�Q�k�C�!H�fff�   ��R�qCz@                                     Bx���d  �          A\)�u�\���(�� C~aH�u��\�{��Cw�                                    Bx���
  
Z          Ap��.{�:=q�p��3C�n�.{�޸R�=q�Cx�                                    Bx���  
�          A�����z����|p�C~&f���@���
=G�Cx��                                    Bx���V  "          A%p��z=q���������\Cr�\�z=q��ff�ڏ\�%p�Co                                    Bx� �  "          A%p��Z�H����\(���ffCyW
�Z�H�����=q��\)Cx5�                                    Bx� �  
�          A%G����\��
=�������Cn����\��\)��
=�p�Ck��                                    Bx� H  
�          A$����{��=q���H�
(�Clp���{�ȣ���ff�"�Ci33                                    Bx� -�  
�          A$z�������z���G��=qCn������������
�/�\Cj��                                    Bx� <�  �          A$(�����������G����Cm�������ff��{�
=Cjc�                                    Bx� K:  
�          A#�
��{��p��Fff����CjQ���{��
=��=q����Ch��                                    Bx� Y�  
�          A#33�����(��B�\���CjJ=�������Q����Ch�3                                    Bx� h�  "          A ����ff��>�p�@\)Cjff��ff��33�
=�g�CjY�                                    Bx� w,  �          A{�ҏ\�޸R�k���Q�Cb���ҏ\��G�����+\)Ca��                                    Bx� ��  
�          A�H��p������   �pz�CYJ=��p���G��L(���Q�CW��                                    Bx� �x  
(          A����
�˅>���?��C[�H���
��33����,��C[�
                                    Bx� �  T          A����\���
?��R@��C[�{��\��
=>��@4z�C\Q�                                    Bx� ��  "          Aff��G��*�H@�=qB3��CK���G��`  @�{B'Q�CQ\                                    Bx� �j  �          A�H���H�
=q@ڏ\B:z�CG�=���H�?\)@У�B/p�CN!H                                    Bx� �  �          A���33�(�@�  BC�\CK���33�Q�@���B6CRW
                                    Bx� ݶ  �          A����  ���@�(�B33CY�
��  ���@���B =qC]�=                                    Bx� �\  "          A���  ���@��\Bz�CY���  ��{@�\)A�p�C\޸                                    Bx� �  �          A�R��\)��z�@�
=A�\)CY�
��\)��@��
A��
C\�                                    Bx�	�  �          A�H������ff@j=qA��C[�)������33@@  A�33C]��                                    Bx�N  �          A���G�����@:=qA��
C_����G��ʏ\@
=qAY�Ca(�                                    Bx�&�  "          A�����
�Ϯ@p�AV�\C`k����
�ָR?�z�Az�Ca^�                                    Bx�5�  T          A  �����@w
=A���C\�)�����33@I��A���C^�f                                    Bx�D@  �          A  ��33��Q�@i��A���C]���33���@<(�A�(�C_�                                    Bx�R�  T          A{�׮��ff@�z�A�=qCZ��׮���@]p�A���C]33                                    Bx�a�  �          A���G���p�?��
@�{C`u���G��У�?�\@I��C`�                                    Bx�p2  "          A  ��z����
<��
=�G�Cb���z���=q�E�����Ca��                                    Bx�~�  �          A	���z����H��
=�P��Ce����z���=q�*=q��G�Cdn                                    Bx��~  T          A\)������
������\Ce������{��z��W�Cd5�                                    Bx��$  "          @�(���\)��p�?�@�
=C`���\)��ff�����C`ٚ                                    Bx���  �          @��������G�>�Q�@6ffCb�������G�����G�Cb��                                    Bx��p            @陚������
=>#�
?�  C_:�������ff��
=�S�
C_#�                                    Bx��  �          @�p�������{���aG�Cc���������Ϳ0����(�Cc�                                    Bx�ּ  �          @������������k�Cc  ������z�.{���Cb�=                                    Bx��b  "          @������H��zᾙ����RCb�\���H��=q�s33��  Cb33                                    Bx��  "          @�  ��G����>�?z�HC`J=��G���z���H�eC`0�                                    Bx��  �          @������R���H��\)���Cb�����R��G��5��=qCb�                                     Bx�T  
�          @�{���R����>8Q�?��C`�����R��z��(��N�RC`�H                                    Bx��  �          @�{���\����>���@p�C_p����\���þ����\)C_p�                                    Bx�.�  �          @�=q���H���;�G��P  C`����H��=q��ff��
=C_�{                                    Bx�=F  �          @�G���z����\�����z�C_^���z����׿Y�����C_
=                                    Bx�K�  
�          @��H���H���?��HAR=qC_\���H����?��A�C_��                                    Bx�Z�  6          @����ff����W
=�У�C^����ff��{�B�\��p�C^0�                                    Bx�i8  �          @�33������
?E�@�=qC^������>L��?�  C^��                                    Bx�w�  �          @�\���H���
>.{?��
Ca#����H��33��
=�L��Ca�                                    Bx���  6          @�\)��33����?�(�AMC^���33���R?�z�A	C^��                                    Bx��*  �          @��
��  ���R@
=AxQ�C\Ǯ��  ���?�ffA6{C]��                                    Bx���  @          @�33������33@p�A��HC\
=�������?�AD��C]@                                     Bx��v  6          @�ff��G�����@��A�ffC\0���G���(�?�{AW�C]}q                                    Bx��  T          A{��p����\@�
A�(�C_=q��p�����?�(�AC�
C`aH                                    Bx���  
�          A{��(����@�A��Ca����(���G�?���AO�
Cc�                                    Bx��h  
�          A���p�����@(�A�  C^����p���  ?�\)AT��C`)                                    Bx��  
�          A�\������{@   A��RCc#�������p�?��AUG�CdJ=                                    Bx���  T          A�\�����\@��A�=qCdY�������?�G�AG\)Ceff                                    Bx�
Z            A��(����\?�33A;�CcJ=��(���
=?��
@��Cc�R                                    Bx�   
�          A ���������@\)A�Q�C_�H������G�?�A]G�Ca)                                    Bx�'�  "          @�������=q@L��A�
=C]�=������(�@+�A��\C_�\                                    Bx�6L  �          @��\��ff��{@Z�HA�p�C[&f��ff����@<(�A��RC].                                    Bx�D�  �          @�(��������@Y��A�\)C[�
������@;�A�z�C]��                                    Bx�S�  "          @�33��p���
=@G�A�C]{��p�����@'�A�=qC^�\                                    Bx�b>  @          @����������@  A��HCb�3�����
=?�
=AD��Cc��                                    Bx�p�            @���\)��G�@XQ�A�Q�C[W
��\)���@;�A�{C]aH                                    Bx��  �          @������H�z�H@q�A�Q�CZ�����H����@W�A�=qC]z�                                    Bx��0  
�          @�z������33@5�A�Q�C]�R������@
=A�
=C_L�                                    Bx���  
�          A�
�����ff@{A{\)CZ�\�������?�(�AA�C[��                                    Bx��|  �          A�R��
=����@<(�A�33C]E��
=����@�A���C^�                                     Bx��"  �          A\)��\)����@VffA��Cak���\)���@333A���Cc                                    Bx���  �          A=q�������@��A��Ca���������@b�\A���Cc�R                                    Bx��n  �          @�����\���@]p�A׮C`޸���\���
@>�RA�p�Cb��                                    Bx��  �          @��
������@QG�A�G�Ca�{������R@1G�A���Cc@                                     Bx���  "          @��H���\��ff@`  Aۙ�Cc0����\����@@��A�z�Cd��                                    Bx�`  �          @���\)��33@'
=A���C^����\)���\@�A��\C`�                                    Bx�  T          @�\)������
=@?\)AÙ�C[޸������\)@$z�A���C]�)                                    Bx� �  �          @�z���(���z�@1�A���C^  ��(���z�@ffA��C_��                                    Bx�/R  �          @�33��  ��33@=p�AŅC^}q��  ���@!�A��C`&f                                    Bx�=�  
x          @���33���
@]p�A�(�Cdc���33��p�@@��A�\)Cf5�                                    Bx�L�  
�          @��hQ����R@z=qB�Chc��hQ����@\��A�CjT{                                    Bx�[D  �          @�Q��w�����@vffB=qCd�f�w����@Z�HA��
Cf                                    Bx�i�  �          @ٙ���(��O\)?fff@�p�CP�H��(��S�
?��@��\CQff                                    Bx�x�  6          @�����XQ�?��HA'�CS�����^{?c�
@�
=CS�=                                    Bx��6  �          @�=q���dz�@��A�\)CW�H���qG�@33A�=qCYff                                    Bx���  �          @�z����R���@n�RA�{Ca�����R��(�@Tz�A߮Cc�                                    Bx���  
�          @����
=���R@Z=qA�Q�C]����
=��  @?\)A�{C_��                                    Bx��(  �          @����H��p�@<(�A�{C]33���H��p�@ ��A��\C^�f                                    Bx���  �          @�ff�����Q�@ ��Av{C_O\�����p�?�ffA=G�C`5�                                    Bx��t  
�          @����R���?���A�HCc�f���R��=q?(�@�z�CdJ=                                    Bx��  �          @������   @�\)B,��CP#����;�@��B"z�CT�                                    Bx���  T          @�ff��Q��c33@xQ�A�\)CU�q��Q��w�@c33A�(�CX\)                                    Bx��f  "          @���p���33@5�A�  C_&f��p����\@��A�Q�C`xR                                    Bx�  �          A Q����\��ff?��HAE�Ce����\�\?�Q�AQ�Cf)                                    Bx��  �          @�z������{@A�A��
C^8R�����@'�A���C_��                                    Bx�(X  �          @�z���=q�u@���B�CY���=q��@s�
A�RC[z�                                    Bx�6�  �          A ����Q��S33@��HB&�RCUc���Q��o\)@�G�B�CX                                    Bx�E�  �          A�H��G����@��BQ�C[����G���Q�@~{A�\)C]�R                                    Bx�TJ  �          A����33��  @g
=A֏\C`k���33����@K�A�\)Cb�                                    Bx�b�  T          A�������
=@�(�B�
C`�������=q@~{A�=qCb�                                    Bx�q�  
�          A ����=q����@mp�Aޣ�Ch���=q��=q@O\)A�z�Cik�                                    Bx��<  
�          @��x�����\@�  B	�Cg#��x�����@uA�Ch��                                    Bx���  �          @������=q@���B=qCd�������@x��A�Q�Cf�                                    Bx���  
x          @�ff���
��33@��B
(�Cc����
��{@��HA�ffCe��                                    Bx��.  h          @�
=��G����@|��A�
=Ce{��G���G�@`��A���Cf�f                                    Bx���  �          @�{�����Q�@qG�A�  Cf��������@U�AɮCg�=                                    Bx��z  T          @�����\)��?�G�AR�RC_aH��\)����?���A ��C`�                                    Bx��   
�          @�p���
=��?��A'\)C_k���
=����?z�H@�C_�                                    Bx���  �          @�(���ff��  @5�A�\)C^s3��ff���R@p�A���C_��                                    Bx��l  �          @�ff��33���R@�A��\Cc
��33���?��HAT��Cc��                                    Bx�  �          @�����{��{@33At  Ce
=��{���\?�{A>�RCe��                                    Bx��  �          @�����
���H@33Aw�Cd�R���
��\)?�\)AB�\Ce��                                    Bx�!^  �          A33��z����R@��A��
C`{��z���(�?�p�A_33C`�                                    Bx�0  
�          A\)������\)@��Ay�Cb��������(�?�\AG�Ccff                                    Bx�>�  �          A�R������
=@A��CeǮ������(�?��AV�RCf�                                    Bx�MP  �          @����{�Å@
=qAz{Chu���{��  ?ٙ�ADQ�Ci
                                    Bx�[�  "          @�(����\���?�\)A>�\Cg#����\���H?�
=A
=qCg�)                                    Bx�j�  �          @�(�������Q�?�AX(�Cd�
������(�?�A&{Cek�                                    Bx�yB  T          @���������?���A:�RCe��������Q�?�AQ�Cf
                                    Bx���  �          A����p����H?�APQ�Cc���p��ƸR?��RA Q�Cc�H                                    Bx���  T          AQ���=q����?��RA!G�Ca����=q�Å?��@��Cbff                                    Bx��4  
�          A(����R��
=?k�@�ffCcs3���R�ȣ�>�@O\)Cc��                                    Bx���  T          A�
�ȣ����>�(�@?\)C\B��ȣ����\=u>�
=C\Y�                                    Bx�  �          Aff��Q���33=�?O\)C]����Q����H��=q����C]��                                    Bx��&  �          A���{���;����Cf�3��{���
�:�H���Cf�=                                    Bx���  �          A�H�������O\)��ffCk  ������H��G��Cj�R                                    Bx��r  
�          Az���=q�ָR��p��%�CjW
��=q��p��Q���{Cj+�                                    Bx��  T          Ap����\�ʏ\@:�HA�Ce����\��Q�@   A�ffCf��                                    Bx��  T          A����R��(�@��
A��\Ci�{���R��{@�{A�(�Cjٚ                                    Bx�d  �          Az���=q���@�G�A��HCj����=q�ᙚ@eA�ffCk��                                    Bx�)
  �          A����33�θR@~�RAљ�Ci{��33��ff@c33A��
Cj
                                    Bx�7�  �          A
=�������R@��
A�ffCe�������\)@\)A��Cg&f                                    Bx�FV  �          AG���
=�θR@�  A�G�Chs3��
=��\)@��\A�(�Ci�{                                    Bx�T�  �          A��
=�Ϯ@��A�  Ch����
=��G�@�z�A��Ci                                    Bx�c�  �          A�H��p���p�@��HA�33Ch����p��ָR@�A��\Ci�                                     Bx�rH  �          A
=��=q��@�{A�z�Cg����=q��ff@�G�A�=qCh��                                    Bx���  �          A�\��G���  @�G�A�Cf�
��G�����@�(�A�{Ch�                                    Bx���  �          A�H��33��  @�(�A�(�Cg����33����@�
=A�z�Ch��                                    Bx��:  "          A����H��  @���A�G�ChB����H�Ϯ@y��A͙�CiO\                                    Bx���  "          A�������(�@9��A�z�Cj.��������@ ��A���Cj��                                    Bx���  �          Az����
��G�?�A�Cg�3���
���
?��\@ָRCg��                                    Bx��,  T          A33���\��  @\(�A�ffCg33���\��{@EA��
Ch\                                    Bx���  �          A���
��=q@���B�C_(����
���H@�Q�B
�C`ٚ                                    Bx��x  T          AQ������qG�@���B0��CZL��������H@�B(p�C\��                                    Bx��  
�          A33��{���@�\)A�33Cf�=��{��\)@��A�p�Cg�                                    Bx��  T          A(���z���ff@�A��Ci�)��z���p�@vffAי�Cj��                                    Bx�j  "          A���ff���@tz�A�{ClQ���ff��  @^{A�\)Cm!H                                    Bx�"  �          A33�z=q��Q�@aG�A�  Cn��z=q��{@K�A��Cn�                                    Bx�0�  T          A����z��N�R@�BOp�CY����z��e�@�  BGC\ٚ                                    Bx�?\  T          A�vff�0  @�(�B]�
CW���vff�Fff@�
=BV��CZٚ                                    Bx�N  T          A�R�q��E@�33BY�C[E�q��\(�@�BR33C^J=                                    Bx�\�  
�          A{�����9��@�BaffCV������Q�@�ffBZz�CZT{                                    Bx�kN  
�          AG��z=q�I��@��Ba��CZ���z=q�aG�@�z�BZ33C^                                      Bx�y�  
�          A\)�x���P��@�(�B]�
C[��x���g�@�RBV\)C^��                                    Bx���  
�          A���~�R�P  @�ffB]��C[5��~�R�g
=@���BVQ�C^+�                                    Bx��@  
�          Az��z=q�Tz�@ڏ\BTffC\L��z=q�h��@��BM{C^��                                    Bx���  �          Az��c33�U�@�{BW�C_0��c33�h��@У�BPG�Ca                                    Bx���  T          A�H��������@�=qA��ChaH�������@���A�{CiW
                                    Bx��2  �          A	G������\)?޸RA;\)Cl@ ����ᙚ?�AQ�Cl��                                    Bx���  �          A	�����ۅ?��HAR�\Cj�q�����ff?�33A0Q�CkL�                                    Bx��~  T          A(�������
=@G�A\Q�Cj�)�����ٙ�?�(�A:�RCj�                                    Bx��$  
�          Az���  �ҏ\@U�A�  Cm.��  ��
=@A�A���Cm�R                                    Bx���  h          A�H��z���  @A�A�=qCik���z����
@0��A�=qCi��                                    Bx�	p  �          A\)�hQ�����@���B�\Cl��hQ����@�G�B\)Cm�3                                    Bx�	  T          A��Mp���\)@��RB�Co�f�Mp���ff@�\)B(�Cp��                                    Bx�	)�  T          AQ��.�R����@У�BPp�Cmp��.�R��G�@��HBH\)Cn�q                                    Bx�	8b  �          A����R��=q@�(�BTG�Cp+���R���H@�ffBL�Cq�H                                    Bx�	G  T          A��
=q���@���BP�Ct�3�
=q���
@��HBGCv�                                    Bx�	U�  �          A=q����G�@��BCyG����Ǯ@�=qBQ�Cy�f                                    Bx�	dT  �          A���{����@�\)BCy�H�{��=q@�
=A��HCz�                                    Bx�	r�  
�          A�������@�G�A��Cz��������H@���A�(�C{J=                                    Bx�	��  
�          A���1G���
=@(�A���Cz�\�1G�� ��@��Aa��C{                                      Bx�	�F  "          A
�R�!G��ff?���@�33C|�=�!G��
=?L��@��HC|�)                                    Bx�	��  
F          AG������
=�G����HC�Q쿬���ff�������
C�N                                    Bx�	��  
n          A�ff��\)�k���=qCG��ff���R��\�a�CB�                                    Bx�	�8  r          Az��J�H��\)?�ffA,Q�Cw��J�H����?��A33Cw+�                                    Bx�	��  
�          A������?L��@��C|ff����?
=q@tz�C|s3                                    Bx�	ل  T          A�R�'
=���\?���A�C{�\�'
=���
?�\)@�C{�f                                    Bx�	�*  
Z          A���\��\)?�G�@�ffC}�R��\�   ?@  @��C~                                    Bx�	��  T          A���{��ff@
�HAk�C~c��{� (�?�z�AO\)C~�                                    Bx�
v  T          A
�\�%�
=?n{@�ffC|n�%��?.{@�Q�C|z�                                    Bx�
  �          A
=q�G
=�   �0����(�Cx� �G
=��
=�p����  Cx��                                    Bx�
"�  �          A	��"�\���\�!�C|�{�"�\�\)��R���C|��                                    Bx�
1h  �          A{�G�� Q�(����C~+��G��   �W
=��Q�C~!H                                    Bx�
@  �          A=q�����p����H�G�C|�
�����zῸQ��{C|�                                    Bx�
N�  �          Aff�p���녿޸R�@  C|���p���Q���H�X(�C|ff                                    Bx�
]Z  �          A(��$z���\���R�_
=C{J=�$z�����(��vffC{(�                                    Bx�
l   "          A  �8Q���  ��{�Pz�Cy  �8Q���{�33�f�HCx޸                                    Bx�
z�  �          A�\�<����R���H�%�Cxc��<����p���33�:�HCxJ=                                    Bx�
�L  �          A ������\�p����C}ٚ����������  C}��                                    Bx�
��  
�          A �����ۅ�[����Cz�=�������e��Q�CzO\                                    Bx�
��  
Z          @�\)�У���\�R�\��33C��׿У���  �]p��ͮC�p�                                    Bx�
�>  "          @��R��{�ᙚ�W���  C��Ϳ�{��
=�a���(�C�z�                                    Bx�
��  �          @�
=��p���=q���(�C�(���p����R����
=C�
=                                    Bx�
Ҋ  
�          @�{�u��G���ff�<�\C��3�u�������Az�C���                                    Bx�
�0  
�          A�H?�\)�ָR��ff��C��f?�\)���
���H�C�ٚ                                    Bx�
��  �          A�?��H��\��p�G�C���?��H��33��R  C���                                    Bx�
�|  �          @�
=<#�
��Q��,(����C��<#�
�޸R�4z���z�C��                                    Bx�"  "          @�Q��
=���
?fff@�
=C�:��
=��z�?G�@��C�<)                                    Bx��  �          @�ff�L���ڏ\@�A�z�C�+��L���ۅ?�z�Ax��C�,�                                    Bx�*n  
<          @�p����\�ٙ�@p�A��C��쿢�\���H@ffA�=qC���                                    Bx�9  
�          @񙚿�{��  ?�Q�Ao�C���{��G�?���Aa�C.                                    Bx�G�  
�          @�p��W
=��(�@,��A�
=C��f�W
=��p�@%A��C���                                    Bx�V`  
�          @���G���G�@7
=A��HC�����G��ڏ\@0��A�(�C�                                    Bx�e  
�          @�33��p���  @�Q�B(�C{��p���=q@�B{C{5�                                    Bx�s�  "          @�=q�33��  @o\)A�(�C|� �33�љ�@i��A�=qC|�H                                    Bx��R  
Z          A Q쿴z��ָR@��A���C����z���Q�@~{A�33C��                                    Bx���  �          AQ���\���
@��A�p�Cz���\��p�@���A�(�C{\                                    Bx���  �          Az��)����G�@��\A�Q�Cw��)�����H@�Q�A��Cx{                                    Bx��D  �          A�H�2�\��=q@���B{Cv(��2�\���
@�\)B�
CvT{                                    Bx���  
�          Az��n{���H@���B  Cm� �n{��z�@�
=B{Cm�R                                    Bx�ː  T          A33���H����@��RB�Cf�����H��=q@��BG�Cf�                                     Bx��6  �          AQ���\)�Ϯ@2�\A�\)Ci����\)��Q�@/\)A�=qCj�                                    Bx���  "          Az���{��ff@=qA��Cj�q��{��
=@ffA�(�Ck\                                    Bx���  T          A���G����
@��A��Ck����G���(�@ffA�ffCk�)                                    Bx�(  �          A�������@"�\A��Ck�����љ�@   A�G�Ck.                                    Bx��  
�          A(���{��
=@%A�Q�Ck����{��\)@#33A�(�Ck�
                                    Bx�#t  �          A����33�У�@-p�A���ClT{��33��G�@+�A���ClaH                                    Bx�2  �          A�\��G���G�@Dz�A�\)Cl��G��љ�@B�\A�Cl�\                                    Bx�@�  
�          A=q��
=����@I��A�=qCm+���
=�љ�@G�A��HCm5�                                    Bx�Of  �          A��p��ҏ\@C33A�G�Cm����p����H@A�A�(�Cm��                                    Bx�^  
�          Ap���=q��@Y��A��RCm�f��=q��{@X��A��Cm�                                    Bx�l�  �          A  ����ff@hQ�A�(�Cm{���θR@g�AɮCm�                                    Bx�{X  T          A��~{�ƸR@xQ�A�(�Cmn�~{�ƸR@w�A��Cmp�                                    Bx���  �          A�������{@|(�A��
Ciٚ������{@|(�A��
Ciٚ                                    Bx���  
�          @�G��s�
����@dz�Aڣ�Cl���s�
����@e�A���Cl�\                                    Bx��J  T          @��[�����@mp�A�CoJ=�[�����@n{A�(�CoB�                                    Bx���  T          @���]p����@u�A�{Cn���]p�����@uA���Cn��                                    Bx�Ė  "          @�ff�N{���@q�A��Cq33�N{��33@s33A��Cq&f                                    Bx��<  T          @��R�:=q��@b�\A�G�Ct�\�:=q���@c�
A��HCt�                                    Bx���  @          @��\�`  �Å@Y��A�Q�Cp(��`  ���H@[�A�{Cp�                                    Bx���  6          @����c33���
@Q�A���Co���c33��33@S�
A���Co��                                    Bx��.  �          @����:�H��=q@j�HA�z�Cu33�:�H�ə�@mp�A���Cu!H                                    Bx��  "          A Q��+���=q@n{A�
=Cw�\�+��љ�@p��A��
Cw��                                    Bx�z  �          A Q��-p���\)@Z=qAə�Cx��-p���ff@]p�A̸RCw��                                    Bx�+   T          A (��:=q��
=@O\)A�
=Cv���:=q��{@S33A�z�Cv��                                    Bx�9�  T          @�p��<(����@S33A�\)Cu�)�<(�����@W
=A�
=Cu�                                    Bx�Hl  
�          @���A���
=@X��A��HCt�f�A���{@\��A��HCt�=                                    Bx�W  
Z          @�
=�=p��Ӆ@Tz�A���Cu�H�=p��ҏ\@X��A�33Cu                                    Bx�e�  �          @�{�?\)��@Dz�A��
Cu���?\)����@H��A�z�Cu��                                    Bx�t^  �          @���6ff��  @'
=A�ffCw�{�6ff�޸R@,(�A�\)Cw�q                                    Bx��  �          A   �l(��ָR@!�A�{Cq(��l(���p�@'
=A��Cq�                                    Bx���  �          A�R�����{@(��A���Cl5��������@.�RA��
Cl�                                    Bx��P  �          AQ����R��\)@#33A��Ckp����R��{@(��A���CkL�                                    Bx���  T          A=q�����p�@!G�A�Q�Cl������(�@'
=A��Ck�R                                    Bx���  �          A\)�g
=�߮@�
A���Cr�3�g
=�޸R@�HA�
=Cr�{                                    Bx��B  �          A�H�o\)��z�@33A��HCqz��o\)��33@=qA�\)CqY�                                    Bx���  T          A���u��ff@\)A�G�Cq{�u����@&ffA�  Cp�                                    Bx��  
�          @�G��a��У�@��A��Cq�=�a���\)@!G�A��RCqc�                                    Bx��4  T          @���y����ff@%A��Cm�{�y�����@-p�A���Cm�H                                    Bx��  
�          @�R�Vff��33@-p�A�{Cq.�Vff��G�@4z�A��Cp�R                                    Bx��  �          @�\� ���ȣ�@5�A�(�Cx.� ���ƸR@=p�A��\Cx                                      Bx�$&  �          @�ff�+��ȣ�@>�RA�
=Cv���+��ƸR@G
=Ař�Cv��                                    Bx�2�  "          @�{�*�H��G�@<��A�
=Cw  �*�H��
=@E�A��Cv�=                                    Bx�Ar            @�=q�-p���p�@4z�A�p�Cv@ �-p��Å@<��A���Cv
=                                    Bx�P  �          @�z��C�
�Å@2�\A��
Csp��C�
����@;�A�
=Cs33                                    Bx�^�  �          @��H�w���  @�HA��
Cl��w���{@#33A��RCkǮ                                    Bx�md  �          @���=q����@G�A~=qCj�R��=q��  @
=qA�(�Cj�q                                    Bx�|
  "          @�G��\)���@ffA�{Ck33�\)��@\)A�\)Cj�3                                    Bx���  �          @�\)��ff��ff@��A�\)Ci�)��ff��z�@"�\A���CiL�                                    Bx��V  T          @������(�@
=qA��Ck
�����=q@�
A�Cj��                                    Bx���  	�          @�(��tz���Q�@!�A���Cls3�tz���{@+�A�
=Cl�                                    Bx���  "          @�
=�u����@A�A�=qCk���u�����@K�AʸRCk=q                                    Bx��H  
Z          @�\)��
=���\@4z�A��HCj
��
=��  @>�RA�p�Ci�3                                    Bx���  
�          @��
��33���@5�A��HCi�f��33���\@@��A��Ci@                                     Bx��  	�          @�G���z�����@Q�A�p�Ch#���z���=q@#33A�=qCgǮ                                    Bx��:  
Z          @�\)�������
?���A(z�Ce!H�������\?�p�A=��Cd��                                    Bx���  
�          @����xQ��Å@:�HA��Cm�=�xQ���Q�@G
=A��
Cm#�                                    Bx��  
�          @�{������@/\)A���Cl#��������H@<(�A�Ck�q                                    Bx�,  
�          @�(���ff���@"�\A��RCk�R��ff��=q@/\)A��CkW
                                    Bx�+�  "          @����Q����
@(�A���Ck!H��Q�����@(��A��Cj�                                     Bx�:x  �          @��R��=q��  @�A���Ck\)��=q���@%�A��Cj�q                                    Bx�I  T          @�ff���R���@�A��Cj����R��=q@   A���Ci�3                                    Bx�W�  �          @��������=q@ffAt(�Ck&f�����Ǯ@�A�p�Cj�\                                    Bx�fj  �          @��
������G�@   Ak33Ck�������ƸR@�RA�\)CkT{                                    Bx�u  
�          @������ff?�p�A3�
Ch�������z�?ٙ�AO
=Ch�3                                    Bx���  �          @�R������z�?��A+33Ch���������H?�ffAF�RCh��                                    Bx��\  �          @��H��33��ff@33ArffCj����33���
@�\A���Cj�)                                    Bx��  
�          @�G������ə�@�\A�z�Cmz�������ff@"�\A��Cm{                                    Bx���  
�          @����G����@z�A�Cl���G�����@33A���Ck��                                    Bx��N  �          @��~{���
@Q�A�z�Cm��~{����@(Q�A�(�Cl��                                    Bx���  "          A�H��=q��ff@!G�A��Cl.��=q���H@2�\A��Ck�R                                    Bx�ۚ  T          A�H��Q���z�@%�A�z�Ckٚ��Q�����@7
=A�z�CkaH                                    Bx��@  �          A����
���@{A�G�CkB����
�љ�@0  A�p�Cj�\                                    Bx���  
�          Ap���ff��(�@=qA�z�Cl
��ff�У�@,��A�
=Ck��                                    Bx��  "          A{��  ��@
=Ap(�Ck��  �ʏ\@��A���Cj��                                    Bx�2  "          A{�������?��RAc33Ck�������@�\A��RCk�\                                    Bx�$�  
�          A �������љ�?���A_�Cl�
������ff@  A�p�Cls3                                    Bx�3~  
�          @��H�x����Q�?��
AAp�Cm��x����?�Ae�Cl��                                    Bx�B$  T          @�
=�vff��\)?�33A~{Cj��vff��(�@
=qA���Cjp�                                    Bx�P�  	�          @ۅ�|������?�33A���Ci#��|����p�@	��A�ffCh��                                    Bx�_p  
�          @�33�e�����@�A�ffCnT{�e���{@�
A�p�Cm�)                                    Bx�n  �          @����33��?�(�Ab=qCh���33���H?�p�A�33Chu�                                    Bx�|�  "          @����z����?���AK\)Ci����z�����?���ApQ�Ci&f                                    Bx��b  �          @�����R��G�?�G�A>{Ci�����R���R?�ffAc�Ci�{                                    Bx��  "          @��
������p�?�33A/�Cj�3�������H?ٙ�AV=qCj�{                                    Bx���  "          @������(�?�\)A*ffCi�������?�
=AQ�Cic�                                    Bx��T  T          @�ff��33��z�?�  A
=Ci�\��33���?ǮAB{Ci5�                                    Bx���  
(          @�{�`����(�?�  Ap�Cp33�`�����?�=qA,(�Co�3                                    Bx�Ԡ  
�          @�{�:=q���
?��RAF�RCt���:=q����?���As�
CtE                                    Bx��F  
�          @أ��<����z�?��A8z�Csp��<�����?�AeCs)                                    Bx���  
�          @ᙚ��=q��ff?.{@��Cjz���=q���?�  A�RCjB�                                    Bx� �  T          @�����Q���G�?�R@�33Chk���Q���  ?p��@�p�Ch33                                    Bx�8  �          @��H��G����>�ff@h��Ch\)��G�����?E�@�  Ch0�                                    Bx��  	�          @���*=q��p�@!�A���Cx���*=q��Q�@;�A���Cx��                                    Bx�,�  �          @�z��.�R�ۅ@0  A��CxO\�.�R��{@I��A�(�Cw��                                    Bx�;*  	�          Aff����p�@8Q�A�p�C{������\)@S33A�\)C{��                                    Bx�I�  
�          AG��p����@7�A�G�C{�H�p���\)@S�
A�p�C{0�                                    Bx�Xv  
(          A��*=q���@C�
A��
Cz  �*=q��33@`  A�{Cy}q                                    Bx�g  
Z          A=q�J=q��@G�A���Cv  �J=q����@c33AȸRCu\)                                    Bx�u�  �          A��C�
���
@7
=A���Cwk��C�
��p�@S�
A�  Cv޸                                    Bx��h  T          A\)�?\)��{@,��A��Cx{�?\)��  @J�HA��Cw�{                                    Bx��  T          A=q�;���(�@.{A�
=Cx^��;���{@K�A�=qCw�)                                    Bx���  �          A���L(���33@7
=A��CuǮ�L(�����@Tz�A���Cu(�                                    Bx��Z  !          Aff�>�R��@(��A��CwE�>�R��p�@FffA�
=Cv�R                                    Bx��   	�          @�\)�<(����@��A�{Cwc��<(���z�@.�RA��Cv�f                                    Bx�ͦ  "          A ���G����
?�
=A^�\CvT{�G���\)@��A�G�Cu�f                                    Bx��L  
(          A���C33���?�z�A=p�CwQ��C33��@	��AvffCv�3                                    Bx���  �          A�Q���\)?�Q�A@(�Cu���Q���33@�Ax��Cu33                                    Bx���  "          A ���*�H��\?��AX��Cz  �*�H��@Q�A��Cy�H                                    Bx�>  �          A{�e��(�?˅A4(�CsB��e��  @�Al��Cr�
                                    Bx��  "          A ���J�H��{?�\AK�Cv33�J�H�ᙚ@G�A��Cu�                                    Bx�%�  
�          @���Mp���?�(�AG\)Cu���Mp���
=@{A�G�CuG�                                    Bx�40  T          @��R�:�H��z�?�
=Aa�Cw���:�H�߮@(�A��HCwY�                                    Bx�B�  T          @�(�����@0  A�C�o\���޸R@P��A�{C�<)                                    Bx�Q|  
�          @����{��G�@(�A��HC��=��{���H@>{A��C��                                     Bx�`"  T          A z�?�(�����@qG�A�z�C�"�?�(���\)@�Q�B��C�q�                                    Bx�n�  T          AG�@�����ff@�B33C��@�����33@�  BffC��                                    Bx�}n  
�          Aff@�33�7�@�p�A�
=C��
@�33�#33@��
BG�C��                                    Bx��  "          A=q@Å�!�@��B�\C��H@Å�
=q@�p�B��C�AH                                    Bx���  "          Aff@���
=@���B�
C���@�����R@�ffB�C��                                    Bx��`  �          A��@����.�R@�ffB{C�S3@����@�z�B=qC��3                                    Bx��  	�          A@����L(�@�33B(�C�B�@����3�
@��HB�\C�˅                                    Bx�Ƭ  
�          A ��@�(��W
=@�B{C���@�(��>�R@�B��C�\                                    Bx��R  �          A Q�@�(��.{@�z�Bp�C��q@�(��@��HB��C�(�                                    Bx���  T          @��@�p��{@�B�C��@�p��
=@��B{C��{                                    Bx��  
�          A ��@��H�(�@��HB�HC�
@��H��
@���BG�C���                                    Bx�D  "          A@��
�Y��@�B��C��R@��
�A�@�{B��C�]q                                    Bx��  
�          A�\@��R�W
=@�B��C�H�@��R�?\)@�{Bp�C���                                    Bx��  T          A�H@��R�Vff@�\)B�
C�U�@��R�>{@��B�RC��                                    Bx�-6  
�          AQ�@�G��K�@��\A���C���@�G��333@��\B�C��)                                    Bx�;�  "          A=q@����G�@�\)A��RC���@���j�H@��B
z�C�R                                    Bx�J�  T          A ��?�p���@�  B �C�5�?�p���G�@���B=qC��                                    Bx�Y(  T          A�?���Ӆ@�ffB\)C�]q?���ƸR@�  B�HC��                                    Bx�g�  �          A �ÿ}p���
=@��B�C��\�}p���=q@�(�B=qC���                                    Bx�vt  
�          A���
=��\)@�p�B�
C�33��
=��=q@�\)BC�{                                    Bx��  "          A   =���33@�(�A��\C�|)=��θR@��RB\)C���                                    Bx���  �          A ��>B�\��@���B��C��\>B�\�ȣ�@��
B=qC��q                                    Bx��f  �          Aff@�ff���H@���A�C���@�ff���R@���B�C��3                                    Bx��  A          A	G�@�p����@��RB  C��R@�p�����@��B�\C�,�                                    Bx���  5          A��@����  @�B�C���@����G�@��\B�C��q                                    Bx��X  T          A33@�ff��Q�@���B��C�S3@�ff��=q@��\B(�C��f                                    Bx���  �          A33@��H��(�@�(�B�
C��=@��H��{@�G�B=qC��{                                    Bx��  "          A=q@�����
@�A�RC�#�@�����R@�ffB=qC�%                                    Bx��J  T          A=q@�  �\@�A���C���@�  ���@�\)B	�C���                                    Bx��  �          A�R@��\��ff@�{A�RC�e@��\����@��RB�
C�xR                                    Bx��  �          A�H@g
=���H@�33A���C�Q�@g
=��p�@��B
�
C�8R                                    Bx�&<  T          A�@g
=���@�\)A�p�C�^�@g
=��(�@���BG�C�O\                                    Bx�4�  "          A��@XQ��˅@���A�C�H@XQ���@��B  C��R                                    Bx�C�  �          Ap�@mp���
=@���A�ffC�j=@mp���G�@�\)B
  C�T{                                    Bx�R.  T          A@����@�\)A�\)C�Z�@�����@�G�BC�ff                                    Bx�`�  
�          A@{���
=@��\A��
C��@{�����@���BffC���                                    Bx�oz  �          A�H@��\�\@�p�A�C��@��\��z�@�  B	G�C���                                    Bx�~   �          A=q@x����=q@z�HA���C��3@x������@���B�\C��R                                    Bx���  T          A@u���H@w�A�ffC��
@u��@�\)B��C�w
                                    Bx��l  �          A{@j=q��G�@��A�(�C�{@j=q���H@���B
�
C�                                    Bx��  �          A�@Y����@�z�A��C��@Y����\)@���B
�HC���                                    Bx���  "          AQ�@"�\����@s�
A�
=C�
@"�\��\)@��BQ�C���                                    Bx��^  "          A�@.�R��p�@n�RA�33C���@.�R��  @�p�B \)C�^�                                    Bx��  T          A�@N�R�أ�@W
=A��HC�� @N�R��z�@�G�A�G�C�g�                                    Bx��            A�
@hQ��Ӆ@S�
A�\)C�aH@hQ���\)@~{A���C��                                    Bx��P  5          A\)@�=q���@S33A��
C�o\@�=q��@|(�A�C�AH                                    Bx��  T          A\)@�G�����@l(�A�33C��@�G����@���A��\C��                                    Bx��  
�          A�@��R��@]p�A�(�C�w
@��R����@�=qA�z�C�n                                    Bx�B  
�          A�@j=q���H@QG�A�\)C��@j=q�ƸR@|��A�C�B�                                    Bx�-�  T          A  @Dz���z�@R�\A�ffC��@Dz���  @�Q�A�
=C��f                                    Bx�<�  T          A(�@Z�H�ٙ�@L(�A�  C�Y�@Z�H��p�@z=qA��C�f                                    Bx�K4  �          A(�@�G��ə�@EA�z�C�q@�G���@p  A�33C��                                    Bx�Y�  "          A��@U���{@^{A���C�=q@U��ȣ�@�p�A�p�C��q                                    Bx�h�  T          A=q@�����@Mp�A��RC�E@����@q�A֣�C�U�                                    Bx�w&  
�          AG�@����(�@HQ�A���C�&f@����  @p��AָRC�R                                    Bx���  �          Az�@�Q��θR@FffA��C��@�Q��\@s33A�ffC���                                    Bx��r  T          A�R@�����@.{A�ffC�K�@������@UA���C�'�                                    Bx��  A          A33@�R��@@��A���C���@�R�ٙ�@r�\A��C�H                                    Bx���  �          A�H@4z�����@1G�A�Q�C���@4z��ٙ�@c33A�Q�C�AH                                    Bx��d  "          A�\@1G���=q@>�RA��C��{@1G���{@p��A�\)C�AH                                    Bx��
  T          A{@�\)�ҏ\@�
Ak�
C�\)@�\)���@2�\A��C���                                    Bx�ݰ  
Z          @�{?�p��߮@QG�A�\)C��\?�p���=q@���A�ffC�\)                                    Bx��V  T          @��H�@  ��@�ffB�C���@  ����@�(�B"33C�<)                                    Bx���  T          @��R��Q����\@��HB'��Cz���Q���ff@�p�B@z�Cw��                                    Bx�	�  T          @��@�\��\)@   A��C�"�@�\��z�@S33A�33C���                                    Bx�H  "          @��@{���  @��
B �C�ff@{���\)@�ffB��C��R                                    Bx�&�  
y          @�33@z=q���@!G�A�  C��H@z=q��
=@O\)AÙ�C���                                    Bx�5�  5          @��@X����p�@6ffA��C��@X������@eA�33C��                                    Bx�D:  
(          @�Q�@aG��ʏ\@333A��C��\@aG���ff@a�A؏\C�S3                                    Bx�R�  �          @��@e���
@#�
A�Q�C��{@e����@S33A��
C�h�                                    Bx�a�  �          @��R@.�R��ff@I��A�C�z�@.�R����@y��A�Q�C�33                                    Bx�p,  �          @�?�p���z�@Q�A���C�N?�p���ff@��A��
C��q                                    Bx�~�  �          @��R@33���@p��A�C���@33����@�Q�B33C��R                                    Bx��x  �          @�G�@Q���(�@qG�A�C�9�@Q����
@���B�HC�f                                    Bx��  �          @�33@.�R��
=@_\)A�p�C�p�@.�R���@�Q�B�C�@                                     Bx���  "          @�{@X����\)@EA�\)C��@X������@w�A�33C��H                                    Bx��j  �          A��@a���(�@@��A���C��@a���ff@s�
A�RC���                                    Bx��  
�          A�\@I����\)@1�A��C�!H@I���ҏ\@hQ�AӅC��f                                    Bx�ֶ  	�          A=q@J�H��\)@0  A�(�C�33@J�H�ҏ\@fffA�=qC�ٚ                                    Bx��\  "          A@1G���=q@4z�A���C���@1G����@k�A�z�C�G�                                    Bx��  T          AG�@^{���@>�RA�G�C���@^{��\)@s33A�ffC��\                                    Bx��  
�          Ap�@C�
��z�@;�A�(�C���@C�
�θR@q�A�33C���                                    Bx�N  "          A��@4z���=q@0��A�33C���@4z����@hQ�A�\)C�}q                                    Bx��  �          A Q�@!G���33@0  A�z�C��@!G���{@h��A�C�Q�                                    Bx�.�  �          A (�@7
=��@2�\A�
=C�8R@7
=�У�@j=qA�p�C��)                                    Bx�=@  "          A ��@:�H��Q�@)��A��C�O\@:�H�Ӆ@a�A�z�C��                                    Bx�K�  T          Ap�@K����
@	��Av�\C�
=@K�����@C�
A���C���                                    Bx�Z�  "          AG�@L(���?�=qAQ��C��R@L(���(�@0  A�p�C�n                                    Bx�i2  
�          A@)����@!G�A���C�@)����33@\��A��C��{                                    Bx�w�  T          A   @/\)��@ffA��C���@/\)�׮@QG�A�p�C�\                                    Bx��~  �          @��@E���p�@�RA�\)C���@E����@HQ�A���C��                                    Bx��$  �          @�{@%���33@��A��C��R@%��ָR@W�A��C��                                     Bx���  "          @�\)@.�R�޸R@0��A��
C���@.�R����@j=qAڸRC�S3                                    Bx��p  
(          @�
=@P���أ�@.{A��C��)@P�����H@fffA�Q�C���                                    Bx��  
�          @��R@Y����=q@8��A��HC���@Y����(�@p  A�\)C��H                                    Bx�ϼ  
�          @�ff@/\)�׮@K�A��\C��@/\)��  @��A�C��R                                    Bx��b  "          @�\)@2�\�ڏ\@?\)A�=qC�)@2�\�˅@x��A�C��
                                    Bx��  T          A ��@����
=@'�A��
C�5�@���ٙ�@e�A��C���                                    Bx���  T          A z�@����@3�
A��C��
@����\)@p��A�\)C�B�                                    Bx�
T  
�          A ��@   ���@S33A�ffC���@   ��  @�\)A���C��\                                    Bx��  "          A{@
�H��@i��Aՙ�C���@
�H�˅@�=qB�C�k�                                    Bx�'�  �          A?�(��޸R@p  A��HC���?�(����
@�BG�C���                                    Bx�6F  T          A{?��R��R@\(�A�=qC���?��R���@��BffC�L�                                    Bx�D�  �          A  ?���ڏ\@�(�B �
C��=?������@���B G�C�!H                                    Bx�S�  �          A���+���{@aG�A�{C�o\�+����
@���B(�C�9�                                    Bx�b8  "          A�ÿ�\)���?���AK\)C�P���\)����@:=qA��\C�f                                    Bx�p�  "          A{���H��(�@Q�A���C��)���H��R@^{A�ffC�q�                                    Bx��  �          A�R�^�R���@j�HA���C��\�^�R��@�ffBC�g�                                    Bx��*  �          A
=�G���  @vffA�z�C�녿G���(�@��HBC���                                    Bx���  T          A�=L����Q�@z�HAܸRC�0�=L�����
@��RB=qC�4{                                    Bx��v  �          A\)���
��  @X��A��C��ý��
��@�
=A��C���                                    Bx��  
�          Aff>������@E�A�Q�C��>�����@�p�A�z�C��R                                    Bx���  
Z          A33?c�
����@J�HA�Q�C�B�?c�
��\)@�Q�A�Q�C��                                     Bx��h  �          A�R?�p����@p  A�ffC��
?�p���G�@�G�B{C�#�                                    Bx��  �          A�R?u����@vffA�  C���?u��Q�@�(�B33C�\                                    Bx���  T          A{?�  ��@|(�A��C��)?�  ����@�ffB�C��                                    Bx�Z  �          Aff?�z���33@s33A��C�p�?�z���
=@��HB��C���                                    Bx�   
�          A�R?˅��=q@u�A��HC��?˅��p�@��
Bz�C���                                    Bx� �  �          A�@,(���33@���A�
=C���@,(����@��B�C���                                    Bx�/L  �          A  @
�H��  @���A�  C��)@
�H��G�@���Bp�C��                                    Bx�=�  T          A
�\?�\)��33@�p�A�\C�&f?�\)����@�
=BQ�C�޸                                    Bx�L�  	�          A
�R?��
���
@�ffA�  C��f?��
���@�  BG�C�w
                                    Bx�[>  "          Az�?����{@���A��
C��f?���ָR@�33B\)C�z�                                    Bx�i�  "          Aff?�ff��\)@�
=A��C��?�ff��
=@�G�B�C��                                     Bx�x�  �          A��?����33@�Q�A�=qC��q?�����
@��B{C�T{                                    Bx��0  �          Az�@   ��z�@��B �C��3@   ��33@�B!�C��q                                    Bx���  
�          A��?�{���H@�  A�{C�q?�{��=q@�=qB��C���                                    Bx��|  
�          A�?�\)��p�@��
A��C�3?�\)��p�@�ffB�RC��
                                    Bx��"  �          A�?�z���ff@��A��C�0�?�z���{@�  BG�C��
                                    Bx���  T          A�?�Q����@�(�A�
=C�P�?�Q���Q�@��B�C�                                      Bx��n  "          A�?��\��(�@���A�Q�C���?��\��(�@��RB�C��                                    Bx��  "          A�\?�=q��z�@�ffA�RC���?�=q��@��
B\)C�L�                                    Bx���  �          A�?��\���@�p�A��C�� ?��\��R@�33BC���                                    Bx��`  "          A
=?�p����@��A��HC�` ?�p���
=@���BC���                                    Bx�  "          A33?}p�� (�@��A�
=C���?}p���\)@�33B{C���                                    Bx��  T          A
=?�
=� ��@���A�\)C�*=?�
=��G�@�\)BG�C���                                    Bx�(R  T          A\)?s33�G�@�=qA��
C�]q?s33�陚@�G�B��C��R                                    Bx�6�  �          A  ?L���Q�@��A�ffC��?L�����@��B=qC��                                    Bx�E�  "          AQ�>�Q��33@q�A��\C�9�>�Q���  @�=qB��C�U�                                    Bx�TD  
�          A
=>L���ff@mp�A���C���>L�����R@�  B�
C��H                                    Bx�b�  �          A  �u���@[�A�C���u����@��RA���C��=                                    Bx�q�  T          A�\����@6ffA�33C�K��������@�z�A��C��                                    Bx��6  �          A����ff��\@�A\)C�����ff���@p  A�ffC�Y�                                    Bx���  
�          A����33�
=@  Ak�C�N��33��
=@dz�A�z�C��                                    Bx���  �          A�H��
=�33@&ffA��C����
=��p�@{�A��HC���                                    Bx��(  T          A\)�Q���H@8��A��RC�8R�Q���33@��RA��\C��                                    Bx���  T          AG��(��Q�@A�A���C��׿(���p�@��\A�
=C���                                    Bx��t  �          A��&ff��
@/\)A���C���&ff��@�G�A�p�C���                                    Bx��  T          Ap��B�\��@K�A��HC�T{�B�\��33@�
=A�C�Ff                                    Bx���  T          A�
?���@eA�\)C��\?�����@�B�C�)                                    Bx��f  "          A33?�  ��R@a�A���C�aH?�  ��\)@��A�C��                                    Bx�  �          Aff?=p���@QG�A�Q�C�z�?=p����\@��
A�C��\                                    Bx��  
�          A�ͽ�Q��z�@?\)A���C��콸Q���p�@��A��C���                                    Bx�!X  �          A
�R�#�
�=q@=p�A��C���#�
��G�@�Q�A뙚C���                                    Bx�/�  �          A�
?B�\��
@\(�A�z�C��?B�\���@�Q�B (�C��H                                    Bx�>�  "          A�\>����@I��A�C��{>���p�@�\)A�{C��
                                    Bx�MJ  
�          A��>�����R@"�\A�ffC�>�����(�@z=qA���C�
                                    Bx�[�  "          A��=u�	�?���AL  C�0�=u��H@W
=A���C�33                                    Bx�j�  T          Ap�?333��@G�AN=qC�Ff?333���@^{A��
C�ff                                    Bx�y<  "          A33@'���\)@��A�\)C��@'���z�@��\B(�C��                                    Bx���  �          A=q@���p�@j=qA���C��
@����@�Q�B
=C�G�                                    Bx���  �          A  ?��R�
{@\(�A��
C�� ?��R��p�@��HA�\)C�                                    Bx��.  "          Aff?��H�
=@I��A���C��3?��H� ��@��\A��HC��                                    Bx���  
Z          A�
?���z�@N�RA�G�C��q?�����@�p�A�{C�e                                    Bx��z  �          A
=?�=q�@<(�A�G�C�G�?�=q��@���Aޏ\C���                                    Bx��   "          A
=?��
�  @Q�A��RC�(�?��
��@�
=A�=qC��H                                    Bx���  �          A
=?�ff�
{@g
=A���C�Ff?�ff��z�@���B ��C���                                    Bx��l  T          A�?�=q���
@�
=A���C���?�=q�߮@�  B  C�Z�                                    Bx��  �          Aff?�����33@��HA�p�C��\?������@�=qB&G�C���                                    Bx��  "          A�\?ٙ����@��A�p�C�4{?ٙ��ڏ\@��
B=qC���                                    Bx�^  "          A�H?�p���(�@��RA�RC�P�?�p���  @�  BG�C��R                                    Bx�)  �          A
=?��H��\)@��A��C�+�?��H��@��
B
=C�Ǯ                                    Bx�7�  �          A�R?Ǯ��\)@�  A��C��f?Ǯ���
@�=qB��C�+�                                    Bx�FP  �          A�?�����@�ffA�=qC�� ?������@�G�Bz�C�'�                                    Bx�T�  �          A
=?�33��\@~�RAͅC��q?�33���H@��HB33C�c�                                    Bx�c�  �          A33?�p��(�@p��A�33C��?�p���\)@�z�B
=C���                                    Bx�rB  �          Aff?������@e�A�  C��3?������@�\)B�C�(�                                    Bx���  �          Aff?���� ��@���A�ffC��q?�����ff@�z�B�C�>�                                    Bx���  T          A��>��ff@(��A�{C�}q>����@��A���C���                                    Bx��4  �          Az�>����@-p�A�C��R>���  @�\)A���C��                                    Bx���  �          A(�>�����@;�A��C�|)>���{@�{A��HC��f                                    Bx���  �          A��<��
�	@_\)A�ffC�{<��
��33@�ffB �C��                                    Bx��&  �          A��?
=�@\)A��
C�H?
=���@��B��C�:�                                    Bx���  �          A��?z��ff@|��A���C���?z����@��
B=qC�0�                                    Bx��r  �          A�?�����@n�RA��C��?����Q�@���B	p�C��                                    Bx��  �          A=q?�ff��
=@�G�A��C���?�ff��\@�z�Bp�C�4{                                    Bx��  �          A�R?���� (�@�  A���C�˅?�����(�@��B�C�`                                     Bx�d  �          A�R?�ff� ��@��RA��HC��3?�ff���@��HB{C�!H                                    Bx�"
  T          A�H?�����@��
A��C���?����@�  BG�C��                                    Bx�0�  
�          A\)?�  �
=@~�RA�p�C�S3?�  ���H@�z�B��C�Ф                                    Bx�?V  �          A\)?���33@~{A�ffC�w
?����@�(�B�C��R                                    Bx�M�  
�          A�R?�33��R@\)A�z�C�?�33��=q@���B\)C�z�                                    Bx�\�  �          A33?�{�(�@xQ�AǮC��R?�{��p�@��B{C�E                                    Bx�kH  �          A\)?h�����@w
=A�{C�"�?h����\)@���Bz�C�|)                                    Bx�y�  �          A�?@  ��\@l(�A�
=C��=?@  ��33@��B(�C��                                    Bx���  �          A\)?˅��\@U�A�{C�]q?˅���@���A�Q�C��                                     Bx��:  �          A�@G��\)@Dz�A�\)C���@G���Q�@���A���C�J=                                    Bx���  �          A��@��	�@A�A�\)C�#�@����
@�G�A��C���                                    Bx���  �          A�H?��H�	G�@U�A��C�q�?��H���\@��HA��
C��                                    Bx��,  �          A�@���	@4z�A�  C�� @����ff@�33A��C�\)                                    Bx���  T          A\)@�
�\)@B�\A�(�C���@�
�   @��HA�=qC�:�                                    Bx��x  
�          A�H@�\�\)@AG�A��C��{@�\�   @�=qA��
C�"�                                    Bx��  �          AQ�@�{@/\)A��C���@��@��\A�{C�#�                                    Bx���  �          A�@
�H���@.�RA��
C��@
�H�ff@���A�=qC�p�                                    Bx�j  �          A�
@Q���@'
=A�C�� @Q���@��RA�z�C�@                                     Bx�  �          A(�@����@333A�
=C���@���H@�z�A�  C�                                      Bx�)�  �          A��@\)�{@,(�A��HC�3@\)��@�G�Aՙ�C��q                                    Bx�8\  �          AQ�@����@"�\Aw�
C���@���@�z�A�Q�C�:�                                    Bx�G  �          A��@�R��@%�Az�\C��3@�R��@�{AϮC�e                                    Bx�U�  �          A�@'
=��@p�AnffC�/\@'
=�(�@�=qA�\)C�                                    Bx�dN  �          AG�@*�H�=q@�Ad��C�XR@*�H���@~�RAģ�C���                                    Bx�r�  �          Ap�@(���@�
A_�C���@(��=q@|��A���C�)                                    Bx���  �          A@ff�Q�@z�A_\)C�O\@ff��H@~{A�
=C���                                    Bx��@  �          A@G��p�@Q�ALz�C��@G��z�@r�\A�{C�s3                                    Bx���  �          A�\@
=�=q@�ABffC�9�@
=�	��@mp�A���C���                                    Bx���  �          A�\@
�H��?�\)A3
=C��q@
�H�\)@e�A��
C���                                    Bx��2  �          A��@Q��
=?�G�A)p�C���@Q��
=@]p�A�G�C���                                    Bx���  �          AG�@p��=q?�ffA-C��\@p��
=q@`  A�G�C�0�                                    Bx��~  �          A��@33�?ٙ�A$  C��@33�	�@X��A�ffC�|)                                    Bx��$  �          A�@#�
���?��A,z�C��=@#�
���@^{A�{C�Z�                                    Bx���  "          A{@W��	��?�=qA�HC��3@W��ff@K�A���C�=q                                    Bx�p  T          Az�@+��\)?��HA%�C�U�@+���@XQ�A���C���                                    Bx�  "          Ap�@%�p�?У�A��C��@%�	�@U�A��\C�Z�                                    Bx�"�  �          A  @!G��Q�?�A��C���@!G��z�@Y��A�\)C�H                                    Bx�1b  �          A  @#�
�Q�?���A\)C��{@#�
���@UA�Q�C��                                    Bx�@  T          A  @{���?�{A(�C�n@{���@VffA�
=C�Ф                                    Bx�N�  �          A��@�H���?��A�C�@ @�H��@Y��A�=qC��H                                    Bx�]T  �          A��@$z����?�(�A!C���@$z����@^{A��C�&f                                    Bx�k�  �          A��@{���?���A.ffC�s3@{�Q�@fffA�z�C��H                                    Bx�z�  "          A�@�R�ff?��
A&�HC��@�R�{@c33A�p�C�                                    Bx��F  �          A��@  �
=?��HA   C��R@  ��H@_\)A�(�C��                                    Bx���  �          Az�@p����?޸RA$  C�l�@p����@`  A��C��{                                    Bx���  �          A  @�H��?��HA:{C�Z�@�H�
�R@l��A���C���                                    Bx��8  �          A33@����?��
A)p�C�B�@���33@a�A���C���                                    Bx���  �          A  ?�33�
=?�  AC��3?�33��@S33A�(�C��q                                    Bx�҄  	          Az�?�p���?�Q�A33C��?�p��(�@O\)A��HC�/\                                    Bx��*  
�          A?�\)�p�?�(�@陚C���?�\)��H@@��A��C��=                                    Bx���  T          A(�?�p���
?�ff@��HC��?�p��@4z�A�(�C�K�                                    Bx��v  �          AG�@ ����R?�(�@陚C��=@ ���(�@>�RA���C�H                                    Bx�   �          AQ�@
�H�33?���@�\)C��H@
�H��@5A�
=C���                                    Bx� �            A  ?����?u@��C��?���H@0  A���C�H�                                    Bx� *h  �          AQ�?�  �=q?�Q�@�ffC��?�  ��@@  A���C�.                                    Bx� 9  �          A��?�  ���?�Q�@�C�XR?�  �{@>�RA��
C��
                                    Bx� G�  �          Aff@#33�\)?�\)A=qC���@#33�(�@H��A�C�R                                    Bx� VZ  �          A=q@-p��ff?��A   C�B�@-p��\)@FffA�Q�C��f                                    Bx� e   �          A�@.�R�=q?�  @�{C�P�@.�R��@@��A��C���                                    Bx� s�  �          A�@'���\?^�R@��C��R@'����@)��A��RC�E                                    Bx� �L  �          A�\@=q�z�?n{@���C�H�@=q��R@.�RA���C���                                    Bx� ��  �          A�\@%�33?��@��RC��)@%�(�@Dz�A��RC�9�                                    Bx� ��  �          A�\@<(��  ?�  A'�C��@<(���@^�RA��C���                                    Bx� �>  �          A��@ff��\?�{A=qC�0�@ff�
�\@XQ�A��HC���                                    Bx� ��  �          A
=@@�����?˅A�C�0�@@�����@UA��
C��                                    Bx� ˊ  �          Aff@��?�@���C�)@��@>{A�Q�C�l�                                    Bx� �0  �          AG�?�\)�ff?L��@���C��f?�\)���@(Q�A�C��                                    Bx� ��  �          Ap�@ff���?�G�@���C�` @ff��\@4z�A�
=C���                                    Bx� �|  �          A��@���  ?xQ�@��HC��3@���{@1�A�G�C��R                                    Bx�!"  �          AG�@33��?xQ�@�33C�H@33�@1�A�33C�J=                                    Bx�!�  �          A�?�\�  ?���@���C�C�?�\�p�@>�RA��HC��                                     Bx�!#n  �          A33@�
��?��@���C�4{@�
�\)@>{A�p�C�z�                                    Bx�!2  �          A��@@  �  ?�A(�C�<)@@  ���@K�A���C���                                    Bx�!@�  �          Ap�@,(��p�?���A��C�<)@,(��
{@J=qA�(�C��                                    Bx�!O`  �          A��@'
=�=q?��@��RC���@'
=�33@E�A��C�W
                                    Bx�!^  �          A�@Q��\)?L��@�
=C�U�@Q��@*�HA~ffC���                                    Bx�!l�  "          A  @�
��
?fff@�=qC�q@�
��@1G�A�(�C�Z�                                    Bx�!{R  �          Az�@���\)?�=q@��
C��@�����@<��A�Q�C���                                    Bx�!��  �          A��@\)��?��@�{C���@\)�G�@:�HA��HC���                                    Bx�!��  �          AQ�@p���?n{@�
=C��3@p��p�@333A�33C���                                    Bx�!�D  �          AQ�@{��?G�@�=qC���@{�=q@*=qA|Q�C��
                                    Bx�!��  �          A��@ff���?
=q@L(�C�33@ff��
@(�Af�HC�e                                    Bx�!Đ  �          A(�@�\�Q�?��@N{C�\@�\��@(�Ag�C�AH                                    Bx�!�6  �          A�?�(����?��@c33C�R?�(���@   An=qC�C�                                    Bx�!��  �          A33?���?+�@~�RC�9�?���
@$z�Av=qC�`                                     Bx�!��  �          A\)?\���?!G�@l��C��H?\��
@!�Aq��C���                                    Bx�!�(  �          A33?��H�z�?�R@i��C��?��H�\)@ ��Ap(�C�AH                                    Bx�"�  T          A�?�33���?�R@k�C��?�33��
@!�Aq�C�                                    Bx�"t  �          A�@ff�33?fff@��\C�C�@ff�G�@1�A���C���                                    Bx�"+  T          A�
?�G����?8Q�@��C�7
?�G��\)@'�AyC�g�                                    Bx�"9�  �          A��?����?J=q@���C�n?����@)��A�33C���                                    Bx�"Hf  �          A�?�=q�(�?fff@�C���?�=q�=q@/\)A�Q�C��3                                    Bx�"W  �          AQ�?���
=?�@K�C�33?���ff@=qAk
=C�N                                    Bx�"e�  �          Ap�?���(�?!G�@p  C�!H?���
=@!�AtQ�C�=q                                    Bx�"tX  �          A�\?�z����?#�
@xQ�C��\?�z���
@   Av=qC��\                                    Bx�"��  �          A=q?�Q����?�R@r�\C��f?�Q���@�RAt��C��f                                    Bx�"��  �          A�\?����G�?!G�@tz�C�B�?����(�@\)Au��C�`                                     Bx�"�J  �          A�?Y����R>�G�@+�C���?Y���=q@�Ad  C��f                                    Bx�"��  �          Aff?W
=���?!G�@qG�C���?W
=�Q�@#33Au��C���                                    Bx�"��  �          A�?G���H>�@5�C�P�?G��=q@�HAf�RC�c�                                    Bx�"�<  �          A�?��\��\?��@N{C��?��\���@�RAlz�C�)                                    Bx�"��  �          A��?��
�33?J=q@��
C�� ?��
�p�@.�RA�G�C��                                    Bx�"�  �          A��?����R?:�H@���C��?���G�@*�HA}�C�<)                                    Bx�"�.  �          A��?�G����>�Q�@
=qC��?�G��Q�@�A[�
C��                                    Bx�#�  T          A(�?G���<#�
=�Q�C�k�?G��z�?��A;33C�y�                                    Bx�#z  �          Aff?�
=�Q�>��@�C�H�?�
=��
@z�A_33C�h�                                    Bx�#$   �          A
=?˅���>k�?�\)C���?˅���@	��AMC�ٚ                                    Bx�#2�  �          A{?�ff��
>B�\?�z�C�� ?�ff�(�@
=AJ�\C���                                    Bx�#Al  �          AQ�?u��=#�
>k�C��?u�(�?���A=p�C���                                    Bx�#P  �          Aff?���ýu�ǮC���?��=q?�\A2�HC�                                    Bx�#^�  �          A  ?n{�\)=��
?�C���?n{�  ?�Q�AB{C��{                                    Bx�#m^  �          A  ?+����L�;�=qC���?+��z�?�A5�C��                                    Bx�#|  �          A�\?z�H����L�;���C��=?z�H��R?��A5�C��R                                    Bx�#��  �          A=q?\(��G��L�;��RC��?\(��ff?�A5�C��3                                    Bx�#�P  �          A  ?aG��
=�W
=��  C��=?aG����?��HA&{C��{                                    Bx�#��  �          A
=?W
=�{���ÿ�p�C���?W
=�  ?�=qAffC���                                    Bx�#��  �          A\)?�
=�{�L�Ϳ��RC��{?�
=��?ٙ�A%��C���                                    Bx�#�B  �          A33?�=q�{���1G�C�E?�=q�z�?���AG�C�N                                    Bx�#��  �          A\)?�
=��\)�X��C��R?�
=�z�?���A\)C��                                     Bx�#�  �          A
=?������9��C�N?���Q�?�
=A\)C�W
                                    Bx�#�4  �          A�?����=q�   �C33C���?������?�z�A��C���                                    Bx�#��  �          A��>#�
��
�n{���C�z�>#�
��?�G�@\C�z�                                    Bx�$�  �          A�=�Q���׿J=q��\)C�C�=�Q��  ?�z�@޸RC�C�                                    Bx�$&  �          A�
�������L��C�����{?�33A�
C���                                    Bx�$+�  �          Az�?:�H���.{���C�8R?:�H��\?�  @�G�C�<)                                    Bx�$:r  t          A�?�\��H�.{��(�C��=?�\�{?��R@�G�C���                                    Bx�$I  4          A��?z��Q�&ff�w�C��q?z��33?�ff@�G�C��                                     Bx�$W�  �          A��>��z�
=�c�
C�e>��33?���AC�h�                                    Bx�$fd  �          AQ�#�
�z����IC����#�
�(�����k�C��f                                    Bx�$u
  �          A�
?E��
=���7�C�W
?E��p�?���Az�C�]q                                    Bx�$��  �          A�?n{��R��33���C��\?n{���?ǮA�
C��R                                    Bx�$�V  �          A\)?�{�녿�\�EC�^�?�{�z�?�33AQ�C�g�                                    Bx�$��  �          A�?�����
�
=q�P  C�O\?����ff?��AC�W
                                    Bx�$��  �          A��?�33�=q���H���C�z�?�33�
=?5@�G�C�u�                                    Bx�$�H  �          Az�?�\)�=q��(����HC�j=?�\)�
=?5@�Q�C�e                                    Bx�$��  �          A�\?fff��Ϳ�\)���C��{?fff�G�?W
=@��RC���                                    Bx�$۔  �          A\)?k��{�J=q��ffC��)?k��p�?�@�p�C���                                    Bx�$�:  �          A{?�ff��׿:�H��=qC�!H?�ff��
?�(�@���C�&f                                    Bx�$��  �          A�?�ff��׿(���|��C�#�?�ff��?��
@���C�(�                                    Bx�%�  T          Aff?Y���p��&ff�uC��f?Y���Q�?��@���C��=                                    Bx�%,  �          A33?z���\���.{C��R?z����?�  A�HC��)                                    Bx�%$�  �          A�\?u�����ff�,(�C��R?u��
?�  A�\C��                                     Bx�%3x  �          A\)?J=q��R��ff�*�HC�S3?J=q���?�G�A33C�Y�                                    Bx�%B  �          A�R?0���{��33�z�C��?0���  ?���A��C�
                                    Bx�%P�  �          A
=?c�
�=q�\�  C���?c�
�(�?���A��C��\                                    Bx�%_j  �          A�?�����8Q쿇�C�G�?����33?�\A'\)C�Z�                                    Bx�%n  �          A�?h����=�\)>��C���?h���@   A@  C��=                                    Bx�%|�  �          A(�@��\)?xQ�@�\)C�5�@���@7
=A���C�xR                                    Bx�%�\  �          A�
@AG��p�?���A*ffC���@AG��z�@l(�A���C�z�                                    Bx�%�  �          A!�@>�R�\)?�(�A�C�� @>�R��R@fffA�p�C�@                                     Bx�%��  �          A!�@9�����?���@��C�t{@9����@QG�A��C�޸                                    Bx�%�N  �          A z�@9����?��
A��C���@9����@Z=qA��C���                                    Bx�%��  �          A z�@@���?�
=A1p�C��@@���z�@qG�A��HC�xR                                    Bx�%Ԛ  �          A!G�@;��Q�?�G�A	�C��=@;��Q�@X��A��C��)                                    Bx�%�@  �          A!@<�����?ǮA��C��R@<���z�@\(�A�G�C��                                    Bx�%��  �          A!@>{�G�?�ff@�(�C���@>{��@L��A��C�                                    Bx�& �  �          A"{@<���(�?�(�Az�C��q@<����@eA��\C��                                    Bx�&2  �          A"=q@C33�Q�?�(�A  C��@C33��@eA�  C�e                                    Bx�&�  �          A"ff@7��{?�A z�C�J=@7��ff@Tz�A��HC��3                                    Bx�&,~  �          A"�\@����?�(�@��
C���@��@J=qA��HC�)                                    Bx�&;$  �          A"�\@�R��?���@�
=C�ff@�R���@P��A��C���                                    Bx�&I�  �          A"�\@z��=q?���@��C��f@z���@A�A��HC�&f                                    Bx�&Xp  �          A"�\?�  �
=?�=q@ÅC�H?�  �Q�@C33A��
C�8R                                    Bx�&g  �          A#\)?�\)� (�?�R@^{C�L�?�\)��H@'
=Al��C�y�                                    Bx�&u�  �          A��?У��33>��@=qC��H?У���H@ffA\z�C��                                    Bx�&�b  �          A{?�ff��?333@��C���?�ff�ff@(Q�Aw33C���                                    Bx�&�  �          A�?ٙ��z�?aG�@��C��\?ٙ���\@3�
A�33C�                                      Bx�&��  �          A ��?�33�p�?E�@��
C�w
?�33��
@-p�Az�RC���                                    Bx�&�T  �          A!�@����?��@�C�E@��ff@AG�A�33C���                                    Bx�&��  �          A{@G���?��A ��C��{@G����@P��A�ffC�AH                                    Bx�&͠  �          Az�@	���\)?���@��C�c�@	�����@?\)A���C���                                    Bx�&�F  �          A  @��?�\)A��C�(�@��
�\@J=qA�33C��                                    Bx�&��  �          A��@   ���?�Q�@�\C�8R@   �
=q@>{A�(�C���                                    Bx�&��  �          AQ�?���(�?�
=@�33C�� ?���p�@?\)A��\C��                                    Bx�'8  �          A��?�G����?Y��@�33C�Q�?�G���
@,(�A���C��f                                    Bx�'�  T          A�?Ǯ��?(��@{�C��?Ǯ�ff@!G�As�C��
                                    Bx�'%�  T          AG�?��R��H?�R@n�RC�y�?��R��@�RApQ�C��                                     Bx�'4*  �          A��?�z���
>\@\)C�9�?�z���@  AX��C�XR                                    Bx�'B�  �          A�?�  ��>�@1�C��f?�  �
=@�Aap�C��                                    Bx�'Qv  �          AG�?�z���H?h��@�p�C�@ ?�z���@0  A��C�j=                                    Bx�'`  �          A��?���=q?�\)@�
=C���?����
@<��A�  C��                                    Bx�'n�  �          Az�?����?��@�(�C�Q�?��
=@=p�A���C���                                    Bx�'}h  �          A?����\)?xQ�@���C��?����G�@4z�A�ffC�<)                                    Bx�'�  �          A�H?�����?@  @�{C��{?�����
@'�A{33C��{                                    Bx�'��  �          A�?��p�?��
@ȣ�C��{?��33@6ffA�(�C���                                    Bx�'�Z  �          A��?s33��?5@���C�� ?s33�ff@#�
Ax��C���                                    Bx�'�   �          A\)?����?�@UC�'�?���G�@(�Aip�C�AH                                    Bx�'Ʀ  �          A�?˅��R?��H@ᙚC���?˅��
@E�A���C�޸                                    Bx�'�L  �          A Q�?��H��?��@�ffC�>�?��H���@Q�A���C�s3                                    Bx�'��  T          A0  A��@��@��BQ�A{�A��@g
=@��B Q�A�G�                                    Bx�'�  �          A0Q�AQ�?�G�@�{B��@�{AQ�@(�@��\A�p�Ar�\                                    Bx�(>  |          A4��A�H?fff@�
=A��@�G�A�H@   @�{A�\A:ff                                    Bx�(�  �          A5�Az�=�Q�@�A�
=?
=qAz�?���@���A��@�\)                                    Bx�(�  �          A4z�A33���R@�  A�\)C�
A33?Q�@�ffA���@�=q                                    Bx�(-0  �          A4��A��aG�@��RA��C�g�A�>�  @���A���?�Q�                                    Bx�(;�  �          A5G�A�׿�G�@�  A�=qC��)A�׾��@�A�
=C��=                                    Bx�(J|  �          A4  A
=��\@��HA��C�A
=�u@�(�A�ffC�'�                                    Bx�(Y"  �          A3�
A�
�0��@��A��
C��A�
��@�G�A�=qC�                                    Bx�(g�  �          A3\)A�Q�@�  A�\)C��A����@��A�Q�C�
                                    Bx�(vn  �          A2�RA��N�R@�A��
C��RA��p�@��RA�z�C�n                                    Bx�(�  �          A2�\A��]p�@��HA�ffC�\A�� ��@�p�A���C���                                    Bx�(��  
�          A1�A  �c33@�ffA�=qC�� A  �!�@�G�A�{C�Y�                                    Bx�(�`  �          A0��A��ff@�z�A�z�C���A�N�R@��A�{C�=q                                    Bx�(�  �          A0��Ap����
@���A�\)C�'�Ap��Z�H@���A�ffC���                                    Bx�(��  �          A0z�A=q���@�z�A�p�C�@ A=q�[�@�z�A�=qC��3                                    Bx�(�R  �          A0Q�A\)����@��
A���C���A\)����@�\)A��C���                                    Bx�(��  �          A/�A�����H@��
A�(�C�  A�����R@���A�p�C�>�                                    Bx�(�  �          A-�A���@�\)A¸RC���A���@��A���C�=q                                    Bx�(�D  �          A,��Aff����@���A�p�C��Aff��@��AݮC�)                                    Bx�)�  �          A+\)@�33��@1G�Apz�C��)@�33�ڏ\@���A��RC��                                    Bx�)�  �          A*ff@�
=��@9��A|��C�w
@�
=�ٙ�@���A�p�C��R                                    Bx�)&6  �          A(  @�=q��@/\)Ar{C�y�@�=q����@���A�  C��H                                    Bx�)4�  �          A%@�����@QG�A�{C���@����
=@�
=Aڣ�C���                                    Bx�)C�  T          A$��@�(��  @-p�At  C��\@�(����H@���A�{C��q                                    Bx�)R(  �          A#33@����@�A@(�C�.@���(�@s33A�33C��                                    Bx�)`�  �          A"�R@Dz���?L��@�Q�C��q@Dz����@&ffAm�C�&f                                    Bx�)ot  �          A"{@U����?�\@8��C�� @U��z�@33AR=qC��                                    Bx�)~  �          A#\)@�Q��33?���@׮C�  @�Q����@7
=A�Q�C���                                    Bx�)��  �          A(��@��H��@�A6�RC�E@��H��{@h��A�{C�*=                                    Bx�)�f  �          A)@���
=?�33A$��C��)@����H@_\)A�{C��=                                    Bx�)�  T          A(��@��H��?�ff@��HC�c�@��H�	�@=p�A��\C���                                    Bx�)��  T          A)�@����G�?ǮA  C�\@����=q@K�A��\C��q                                    Bx�)�X  �          A(��@�  �{?��H@�ffC��f@�  �33@EA�Q�C��=                                    Bx�)��  T          A*�R@ʏ\���?��@<��C�AH@ʏ\���@Q�A8��C��=                                    Bx�)�  �          A��@(���p�@�33A�  C�Ф@(���@�ffB��C�f                                    Bx�)�J  �          A$��@L���  @l��A���C��@L����\@��A��
C���                                    Bx�*�  �          A�
@>�R�Q�@aG�A���C�` @>�R��
=@��HA�33C�@                                     Bx�*�  �          A{@n{��@Tz�A�  C��
@n{���R@��HA뙚C��q                                    Bx�*<  �          A!@�33��
@=qA\z�C��H@�33����@x��A���C�|)                                    Bx�*-�  T          A!�@�\)�   @;�A�z�C�4{@�\)��=q@�33A�C�g�                                    Bx�*<�  T          A\)@�G���\@C33A�33C��@�G���ff@�G�A�(�C��\                                    Bx�*K.  �          A�H@`���=q@��Al��C�W
@`������@y��A�(�C�R                                    Bx�*Y�  T          A33@_\)�	�@+�A���C�@_\)��\)@�
=A���C��                                    Bx�*hz  �          A�@p����
?�\A)�C���@p���(�@UA�C�9�                                    Bx�*w   �          A��@s33�z�@33AAp�C��3@s33�(�@g�A���C�Y�                                    Bx�*��  �          A\)@g����?��
A)p�C�*=@g��p�@W
=A�=qC���                                    Bx�*�l  �          A�
@z��{?��
@�p�C���@z����@(Q�A���C���                                    Bx�*�  T          A(�@h���ff?�33@��
C��@h�����@'�A��C��                                     Bx�*��  �          A\)@/\)��>�\)?�{C�9�@/\)� ��?�\A<  C�l�                                    Bx�*�^  �          A��@|����p�>��?�C�>�@|����  ?�Q�A/�C��H                                    Bx�*�  �          A�@�{����>�z�?�{C���@�{��
=?��HA0z�C�AH                                    Bx�*ݪ  �          A�@�������>\@Q�C��\@�����ff?��A6�RC�޸                                    Bx�*�P  �          Aff@�������>aG�?���C�Ф@������
?�{A"=qC�R                                    Bx�*��  �          A��@�\)���ý�Q�z�C��f@�\)��p�?�ffA�
C��{                                    Bx�+	�  �          A=q@�p���33��  ��=qC�c�@�p���Q�?�z�@��C��                                    Bx�+B  �          A�
@���=q�=p���33C���@���=q?8Q�@��C��{                                    Bx�+&�  �          AG�@��R����z�H��ffC��f@��R� ��>�@8Q�C���                                    Bx�+5�  �          A��@��R�ڏ\���H�P  C���@��R���?:�H@�z�C��3                                    Bx�+D4  �          A��@^�R��������p�C�Ф@^�R���u��\C�U�                                    Bx�+R�  �          A��?k����H��=q� 33C�Ф?k����B�\���C�|)                                    Bx�+a�  �          A�
@�
��p������1�C�c�@�
��=q�L�Ϳ���C�<)                                    Bx�+p&  
�          A�@\)���.{��33C�S3@\)��33������RC���                                    Bx�+~�  �          A  @I����{���MC�|)@I�����Ϳ�\�a�C�4{                                    Bx�+�r  �          A�@��\�����ff�j�\C�L�@��\����\(���C���                                    Bx�+�  �          A(�@Q���(����
��ffC���@Q����7
=���C�Ф                                    Bx�+��  �          A��@mp��ָR�Z=q��C�t{@mp���R�(��qC��q                                    Bx�+�d  �          A
�\@��\��Q���\�^=qC��)@��\��Q�J=q���\C�k�                                    Bx�+�
  �          A\)@^�R���ff�_�C�0�@^�R���:�H��33C�ٚ                                    Bx�+ְ  �          A��@HQ���p��
=q�d(�C��@HQ���H�B�\���RC�w
                                    Bx�+�V  �          A@W
=����������C��H@W
=�������p�C�<)                                    Bx�+��  �          A�H@~{��G��o\)��z�C��3@~{��\�{���RC�Ф                                    Bx�,�  �          A�@���ff��G�����C��
@���G��5�����C��                                    Bx�,H  �          Az�@H����
=�W
=��ffC�j=@H������\�V�RC��\                                    Bx�,�  T          Ap�@C�
���H�R�\��33C��R@C�
� �׿�Q��K�
C�h�                                    Bx�,.�  
�          A
=@S33���H�S33���C���@S33� �Ϳ��H�K
=C�"�                                    Bx�,=:  �          A@i����33�S�
��C�9�@i������� ���RffC���                                    Bx�,K�  �          A(�@A�����N{��z�C��@A��������G�C�e                                    Bx�,Z�  �          A��@P  ��33�?\)��G�C���@P  ������.�HC��                                    Bx�,i,  �          A��@S�
����1G����RC���@S�
� Q쿸Q��ffC�33                                    Bx�,w�  �          A�R@�\)��ff�E��(�C��=@�\)��z���\�^=qC���                                    Bx�,�x  �          A
=@ָR����z��V�RC�W
@ָR��Q쿏\)��RC��=                                    Bx�,�  �          Az�@�
=���ÿ����C��@�
=���Ǯ�$z�C��{                                    Bx�,��  �          A
=q@{���(��   �V�HC��@{����=p���C���                                    Bx�,�j  �          A
�\@�����=q��\)�-C�!H@������Ǯ�$z�C�ٚ                                    Bx�,�  �          A�@_\)��{�}p���  C��R@_\)�   >�{@\)C��f                                    Bx�,϶  �          AG�@R�\�녿�\)��G�C��@R�\�
=>u?��
C��                                    Bx�,�\  t          A��@I���
=��=q����C���@I���(�>�\)?���C�p�                                    Bx�,�  4          A��@I������
��C�T{@I���\)=�?:�HC�8R                                    Bx�,��  �          A@E������Q��EG�C�"�@E���Ϳ���^�RC���                                    Bx�-
N  �          Aff@_\)�
=�Q��W
=C���@_\)��H�B�\����C�@                                     Bx�-�  �          Az�@�
=��  �aG����
C�J=@�
=��\)��p(�C�p�                                    Bx�-'�  �          A
=@�G���G��W
=���C���@�G������b�RC��                                    Bx�-6@  �          A=q@xQ���G��r�\��(�C�k�@xQ�����&ff��ffC��3                                    Bx�-D�  T          AG�@�Q���z��  �G�C�ٚ@�Q���G��u����C���                                    Bx�-S�  �          A��@^{�����  �(�C���@^{�{�L�Ϳ��\C��                                    Bx�-b2  �          A
�\@_\)���׿����'\)C��@_\)��p����R��\C��f                                    Bx�-p�  �          A	@J=q�����|��C�33@J=q��{������C�ٚ                                    Bx�-~  �          A�
@9����{�.{��ffC��f@9�����ÿ�  �$  C�9�                                    Bx�-�$  �          @�@#33�θR�o\)��ffC�@#33��
=�*�H��(�C��                                    Bx�-��  
�          @�p�@-p���  �Z=q����C�%@-p���
=��H��{C�]q                                    Bx�-�p  �          @��
@W���33�����(�C���@W����Ϳ����"�RC�                                    Bx�-�  �          AG�@�Q���G���Q��	p�C�H�@�Q���z��G��J=qC�R                                    Bx�-ȼ  �          A��@vff����2�\��
=C�J=@vff�����33�3
=C��
                                    Bx�-�b  �          A�
@U��\)�\(����HC���@U����>�33@��C���                                    Bx�-�  �          A�\@Vff��׿�=q�33C��)@Vff��z������
C��3                                    Bx�-��  �          A�@Z=q���H�~{��33C��q@Z=q��(��>�R��ffC���                                    Bx�.T  T          A33@S33�������\��C��3@S33�����XQ���p�C��                                    Bx�.�  �          A	p�@���ۅ�
=q�j�RC��@������=q��G�C�xR                                    Bx�. �  �          A
�\@�  ��Q��P������C�R@�  ���(��j�HC�S3                                    Bx�./F  �          A�@����������
��  C�)@����߮�U���
C��q                                    Bx�.=�  �          A  @�G��ƸR���
���C�L�@�G�����XQ���=qC��                                    Bx�.L�  �          AQ�@n{�������\�{C�O\@n{��{�u���p�C�
                                    Bx�.[8  �          A��@g
=�������R��C�y�@g
=����j�H���HC�`                                     Bx�.i�  �          A	�@G
=���R��G��  C��@G
=�����H���C�u�                                    Bx�.x�  �          A\)@\)��z���z�� p�C�o\@\)��(���ff� p�C�E                                    Bx�.�*  �          A�@c�
�����������C��@c�
����^�R���C���                                    Bx�.��  �          AG�@}p��љ�������C���@}p���ff�u��  C�h�                                    Bx�.�v  T          Az�@fff��=q��Q�����C��H@fff��ff�j=q��z�C���                                    Bx�.�  �          A��@Vff��  ��=q�ݙ�C�f@Vff� ���J=q��\)C�H�                                    Bx�.��  �          A=q@s33���H����p�C���@s33����S33��
=C���                                    Bx�.�h  �          A�@y����ff��Q����C�:�@y����=q�i����p�C�B�                                    Bx�.�  �          A�R@`�����H��  ��{C��f@`����ff�hQ���33C��                                    Bx�.��  �          A=q@c�
����������
C�8R@c�
��G��s�
���C�Ff                                    Bx�.�Z  �          A33@XQ��ᙚ������=qC�Ф@XQ���p��n�R���
C��f                                    Bx�/   �          A��@}p���  �dz�����C�Q�@}p���{�\)��ffC��                                     Bx�/�  �          Aff@u����������{C��H@u���\)�^�R���C���                                    Bx�/(L  �          A��@Mp���  ��=q� Q�C�P�@Mp����q��ď\C�k�                                    Bx�/6�  �          A��@�������H��HC���@�����
�������
C���                                    Bx�/E�  �          AQ�?�Q���  ����� {C��?�Q���  ����� 33C�n                                    Bx�/T>  T          A�@z���\)�������C�0�@z����H�qG��˅C��                                     Bx�/b�  �          A�\@(����=q��G���
=C��@(����33�N�R��(�C�Ff                                    Bx�/q�  �          A\)@
=���H��ff��Q�C�@
=����Fff��Q�C�K�                                    Bx�/�0  �          A(�@�H��
=��(���G�C���@�H� (��S33��Q�C�j=                                    Bx�/��  �          AQ�@1G���\)���H��\)C�xR@1G������c33���C��                                    Bx�/�|  �          A(�@�R��=q��33��(�C�\)@�R��z��c33��  C��)                                    Bx�/�"  �          Az�@(���\)�����{C�
=@(�� (��U��  C�xR                                    Bx�/��  �          Aff@3�
������C��=@3�
���R�x�����HC�                                    Bx�/�n  �          A33@5��Q���z�� z�C��3@5��33�w
=�Ə\C��3                                    Bx�/�  T          A��@*�H���
������C���@*�H�ff�Z�H���
C�{                                    Bx�/�  �          Aff@`  ��\)�a���z�C���@`  �{���o\)C�XR                                    Bx�/�`  �          A��@H�������g
=���HC�&f@H��� ���#�
���C��H                                    Bx�0  �          A33@S33������G���\)C���@S33��33�s�
�ŅC���                                    Bx�0�  �          A{@:�H�Ǯ��Q����C���@:�H��{��(��C�ff                                    Bx�0!R  �          A=q@!G����H��=q����C�|)@!G�����s�
��33C��R                                    Bx�0/�  �          A�@j�H�����g���
=C��{@j�H� ���%��ffC�AH                                    Bx�0>�  �          A�@����Ǯ����\)C�` @�����������p�C��                                    Bx�0MD  �          A�@�
=��(��l����G�C��q@�
=���/\)��
=C���                                    Bx�0[�  �          Aff@��
��(��a���Q�C��@��
�����"�\�{33C�Z�                                    Bx�0j�  �          A�@�����K���ffC�� @�����\�(��U�C��                                    Bx�0y6  �          Aff@��
��R�S�
��G�C�� @��
��=q���eC�Ff                                    Bx�0��  �          AG�@W�������H���HC��@W���\)�Fff��G�C�s3                                    Bx�0��  �          A��@�
=��Q��,�����\C�%@�
=�ᙚ�����>�\C���                                    Bx�0�(  �          A�H@�ff����(������C�@�ff����
=�-�C�Q�                                    Bx�0��  �          A  ���
�4z�����R(�CS\)���
�k�����A��CZ��                                    Bx�0�t  �          A�H��ff��H���R�^�COB���ff�W
=���H�O�CW�{                                    Bx�0�  �          A�\���R�(Q�����l�\CT���R�h����p��[��C\޸                                    Bx�0��  �          A33��ff���H��g�CJ����ff�<�������Zp�CT�                                    Bx�0�f  �          A����R�xQ����^  C>����R������z��VCG�H                                    Bx�0�  �          A����G�=�G���
=�X  C2���G��\(���p��V33C<}q                                    Bx�1�  �          A=q��\)?=p���R�L\)C,����\)�.{���M��C5��                                    Bx�1X  �          A
{?�33��=q�XQ�����C�1�?�33��R�=q��=qC�                                    Bx�1(�  �          A  ?}p���p��������C��?}p�����g
=���HC���                                    Bx�17�  T          A(�@ff��z�������G�C�Ф@ff��=q�G
=��
=C�\)                                    Bx�1FJ  �          Aff?�����\)�^�R��Q�C�n?����G�� ����33C�*=                                    Bx�1T�  �          A?�����J=q���C���?�����H���d(�C�\)                                    Bx�1c�  �          A��?���33�����(�C�� ?����;��
��C�s3                                    Bx�1r<  �          A  ?��H�  ���=qC��?��H�	��
=�1G�C�                                      Bx�1��  �          Ap�?^�R����z��G�C�t{?^�R��ff�������
C�9�                                    Bx�1��  �          Az�>�G����n�R��\)C���>�G�����2�\����C���                                    Bx�1�.  �          A
�H>�ff����Q��ep�C��=>�ff��
��z�����C��H                                    Bx�1��  �          A33?�����H��Q�����C�B�?���   �HQ���G�C��                                    Bx�1�z  �          A=q?������������C�N?�����p��y����  C�
=                                    Bx�1�   �          Ap�?��H�陚������C��)?��H���
��ff��{C�#�                                    Bx�1��  �          A{?ٙ���Q���p��  C��
?ٙ����H��=q��{C�)                                    Bx�1�l  �          Az�?��H��  ���'G�C��?��H��z���z��ffC��
                                    Bx�1�  �          A�?�ff����33�%z�C���?�ff������\�
=C��                                    Bx�2�  �          A=q?�(����\�-p�����C�/\?�(��G����I�C�                                      Bx�2^  �          A�@4z���Q��0����C���@4z�� (���33�IC���                                    Bx�2"  �          A�
@~�R��R�8Q����\C��@~�R��\)��
�U��C���                                    Bx�20�  �          A�@U�����e���=qC��@U�����0�����C�xR                                    Bx�2?P  �          A  @   �����u����C�33@   �=q�@  ��p�C���                                    Bx�2M�  �          Az�@�
��ff��������C���@�
��(��mp��\C�&f                                    Bx�2\�  �          A��@  ��\��Q�����C��@  ��G��~{�иRC��                                    Bx�2kB  �          A�@ff��������p�C��@ff��\)�xQ���G�C�c�                                    Bx�2y�  �          A=q@\)���H��\)��G�C�\)@\)��G��|����(�C���                                    Bx�2��  �          A(�@P  ��33�L(���
=C���@P  ��z�����{33C�/\                                    Bx�2�4  �          AQ�@Z�H����'���{C���@Z�H�p����:=qC�s3                                    Bx�2��  �          A�\@Vff�ٙ���
=��C��@Vff�����Q���Q�C�]q                                    Bx�2��  �          A��@e������z��*(�C�*=@e���  �����p�C��=                                    Bx�2�&  �          A@K�������(��G�C��f@K�������  ��HC��R                                    Bx�2��  �          Az�@;����;��R�z�C�/\@;���z�>�@O\)C�1�                                    Bx�2�r  �          A\)@��陚���R�=qC��@�����E���=qC�s3                                    Bx�2�  �          A  @O\)��\�QG���
=C��3@O\)����!G����
C�33                                    Bx�2��  �          A�R@I�������33�X��C�� @I��� �ÿ��
��\C���                                    Bx�3d  �          A�@��������������C���@����θR�{���G�C��)                                    Bx�3
  �          AQ�@j=q����޸R�N��C���@j=q��Q���G��=�
C��\                                    Bx�3)�  �          Ap�@o\)�x�������UC��@o\)�����(��E�\C���                                    Bx�38V  �          A(�@8Q�� ���=qǮC�p�@8Q��P�����
�t\)C��3                                    Bx�3F�  �          A�@!G������ffC�W
@!G������=q.C�N                                    Bx�3U�  �          Az�@=q=�G��
�R�H@#33@=q�0���
=q��C��                                    Bx�3dH  �          A�@���Q����8RC��)@��\(��  8RC�4{                                    Bx�3r�  �          A�@C33@O\)��33�q��B:�H@C33@ ����B                                      Bx�3��  �          Az�@3�
�7
=�����a\)C�7
@3�
�X������R
=C��                                    Bx�3�:  �          A\)@C�
���R��  �C�  @C�
��33���
�C�
                                    Bx�3��  �          A��@>�R�xQ���{�_{C��@>�R��  ��=q�N��C��                                    Bx�3��  �          A�@`��������S�
C�p�@`�������\)�D=qC�~�                                    Bx�3�,  �          Ap�@���������'=qC�n@�������G��\)C��                                    Bx�3��  �          Ap�@n�R�`����z��`�C�` @n�R��������R��C��R                                    Bx�3�x  �          A
=@�R?����\��A��@�R?�\���A8��                                    Bx�3�  �          A{?�
=��=q��33u�C�Ф?�
=��Q���\)#�C�s3                                    Bx�3��  �          Aff��33��
�Z=q��z�C��R��33�  �*�H��(�C��H                                    Bx�4j  �          Ap�����\)�~{��  C��f����(��O\)��{C���                                    Bx�4  �          A�\?�R�  ��
=��\)C��?�R�p��p�����
C�                                    Bx�4"�  �          Ap�=L������o\)��=qC�%=L�����@������C�#�                                    Bx�41\  �          A�\��G�������\��33C��f��G��{�hQ����\C��                                    Bx�4@  �          AG���\�Q���G���{C�8R��\�	�����˙�C�J=                                    Bx�4N�  �          A�����(��Ǯ� �RC��=����33��(���C���                                    Bx�4]N  �          A
=��\)�G���Q��  C��q��\)�  ���
��=qC�
=                                    Bx�4k�  �          A33>���p��������C�w
>����������C�ff                                    Bx�4z�  �          Az�?#�
���������C�^�?#�
����p��z�C�>�                                    Bx�4�@  �          A�>k���=q���H�(�C�� >k�� (���  �	��C��3                                    Bx�4��  �          A{?!G���G����G�C�b�?!G����R��33�33C�B�                                    Bx�4��  �          Az��_\)��������,�Co:��_\)��=q��ff���Cq
                                    Bx�4�2  �          A�H�/\)�����Q��5p�Ct���/\)��������&�Cvp�                                    Bx�4��  �          A  �   ����Q��,=qC}\)�   ��(���Q��G�C~^�                                    Bx�4�~  �          A{�У��љ���ff�.��C�q�У��߮���R��
C�l�                                    Bx�4�$  �          A=q��=q��\)��33�?�HC~�\��=q��{���033C��                                    Bx�4��  �          Aff��
=���
��33�G{C�ῷ
=�\��{�7z�C�e                                    Bx�4�p  �          AQ�?
=��p���33�%=qC�s3?
=��=q����p�C�P�                                    Bx�5  �          Aff?5�ٙ���=q�&\)C���?5��ff���H�C�˅                                    Bx�5�  �          A�
?z���(���Q���\C�>�?z������  ��C�#�                                    Bx�5*b  �          A��?��R��z����R��=qC�t{?��R��H��p����C�K�                                    Bx�59  �          AQ�?�{�������\)C��f?�{�ff�����ՅC���                                    Bx�5G�  �          A  ?��
��ff�������RC���?��
�   ������p�C�ff                                    Bx�5VT  �          A�?���\)��p���C�c�?���������Q�C�,�                                    Bx�5d�  �          A�H?�p���\)����  C�h�?�p�� (�����\)C�9�                                    Bx�5s�  �          A��?�\)���������C�@ ?�\)��G���p�����C�
                                    Bx�5�F  �          A��?��\��(���33�=qC�4{?��\��ff����33C�                                    Bx�5��  �          AG�?��R�  ���
�5�C��{?��R�p����R���C�Ǯ                                    Bx�5��  �          AQ�>L����\�Dz�����C���>L���	G��#�
��
=C��                                    Bx�5�8  �          A��aG��Q��^{��(�C�9��aG��\)�>{���HC�=q                                    Bx�5��  �          AG�>W
=��R�W
=��  C���>W
=�	���7
=���C��
                                    Bx�5˄  �          A33>�G���=q�o\)��(�C���>�G�� z��Q����C��{                                    Bx�5�*  �          A��?����
=��\)��  C�,�?����
=������C��                                    Bx�5��  �          Aff?!G���=q�\�=ffC�8R?!G�������Q��0�\C�
=                                    Bx�5�v  �          A�
?�Q������H��HC���?�Q�����{����C���                                    Bx�6  �          A(�@Q���������
=C���@Q���{���R�Q�C�&f                                    Bx�6�  �          A�\@�������
=�Q�C���@����z����R�33C��R                                    Bx�6#h  �          A�\@���  ��(���\C�@���  ����\)C��                                    Bx�62  �          A	p�@�33���
����tQ�C��@�33��\)��33�Lz�C���                                    Bx�6@�  �          A	@�z���\)��G���Q�C�}q@�z����ÿ&ff���HC�h�                                    Bx�6OZ  �          A�@�������:=q����C�g�@��������#33��{C�&f                                    Bx�6^   �          Ap�@�(���\�>�R���C��q@�(���
=�'���{C���                                    Bx�6l�  �          A��@vff����n{��\)C�` @vff��ff�W����C��                                    Bx�6{L  �          A�@c�
���P  ��ffC��@c�
��=q�8����z�C���                                    Bx�6��  �          A@r�\�陚�N�R��G�C���@r�\��ff�8Q���  C��                                     Bx�6��  �          A��@dz���\�'�����C�� @dz���{�G��o\)C�u�                                    Bx�6�>  �          Aff@Z=q�=q�n{��G�C�]q@Z=q��H�\)�hQ�C�S3                                    Bx�6��  �          Aff@`  ���������C���@`  �녿u��{C���                                    Bx�6Ċ  �          Ap�@��H��\�E���C��@��H��R�1���=qC���                                    Bx�6�0  �          A��@J=q�G����Q�C���@J=q�{������
=C��                                     Bx�6��  �          A�@Q����Ǯ�"�RC�\@Q��{��(���C��q                                    Bx�6�|  �          A�@G
=� �ÿ��.�RC���@G
=�{��=q�\)C�xR                                    Bx�6�"  �          A��@p���\������p�C�'�@p���\=#�
>��C�&f                                    Bx�7�  �          A��?�  �
�H��  ��{C�L�?�  �
==���?(��C�K�                                    Bx�7n  �          A��?.{��?�@Z�HC�=q?.{�33?\(�@���C�@                                     Bx�7+  �          AQ�?���{?��
A	C�b�?����?���A,Q�C�j=                                    Bx�79�  �          A�@:�H���\@p�A�
=C�7
@:�H���@0  A��C�T{                                    Bx�7H`  �          A��@G���\)�#�
��p�C���@G���  ��Q���RC��3                                    Bx�7W  �          A\)@�R�������/\)C��R@�R�G���G��E�C���                                    Bx�7e�  �          A@I�������Q�(�C�*=@I�����>8Q�?��RC�+�                                    Bx�7tR  �          A=q@n�R����=q��=qC�P�@n�R��{�#�
��Q�C�O\                                    Bx�7��  �          A\)@��H��=q��=q�HQ�C��
@��H��(������/
=C���                                    Bx�7��  �          A  @�Q���p��p����\C��@�Q��߮��R�s�
C��=                                    Bx�7�D  �          A�\@���  �*=q��p�C��H@���\��H����C���                                    Bx�7��  �          AQ�@����G��=q��Q�C�  @��������pQ�C��H                                    Bx�7��  �          A��@���  ���m��C�]q@���녿�(��W
=C�>�                                    Bx�7�6  �          A��@�����H��G���C�^�@�����
��ff��\C�N                                    Bx�7��  T          A��@�(�����ff�C33C��@�(���\)��{�-C�˅                                    Bx�7�  �          A	�@����ff��Q���RC��@���׮��  ��RC���                                    Bx�7�(  �          AQ�@�����  ��\)�/\)C���@�����G���Q���C��                                    Bx�8�  �          A��@����Q�5����C�@ @�����ÿ���l(�C�7
                                    Bx�8t  �          A	��@�(���p��ٙ��6�\C��=@�(��ָR���
�$z�C���                                    Bx�8$  �          A
=q@�G����Ϳ����E�C��@�G���ff��z��3�C�xR                                    Bx�82�  �          A
ff@�33��{���\�\)C�O\@�33��
=������\C�B�                                    Bx�8Af  �          A
�\@��
��ff��{��z�C�T{@��
��
=�s33���C�J=                                    Bx�8P  �          A33@�33���
���H��
=C�  @�33���Ϳ�ff��ffC�3                                    Bx�8^�  �          A�
@����ָR��\�t  C�o\@�����Q��	���d��C�W
                                    Bx�8mX  �          AQ�@��R��
=��=q��G�C��q@��R�����|����=qC��                                    Bx�8{�  �          A	@�(��ҏ\�S�
��  C��3@�(���z��L(����RC��3                                    Bx�8��  �          A	@�p���{�[����C��R@�p���  �S33����C�ٚ                                    Bx�8�J  �          A(�@�(���=q��33��  C���@�(���z���  �ۮC��                                     Bx�8��  �          A�R@�z����\��\)��C��
@�z���������p�C��3                                    Bx�8��  �          A�
@�(��]p����
�;�C�U�@�(��c�
����9  C��3                                    Bx�8�<  �          A
ff@g���
=����}  C��3@g���ff��G��{�HC�&f                                    Bx�8��  �          A
=@�33��׾�
=�.{C���@�33���þ��
�z�C��                                    Bx�8�  �          A��@�\)���>�(�@,��C��H@�\)���?�@UC���                                    Bx�8�.  �          A��@��R���H?(�@vffC��\@��R��\?333@�{C��3                                    Bx�8��  �          A  @�(���>.{?��C�o\@�(���>��?�33C�p�                                    Bx�9z  �          A(�@�����>\@�C��@�����>�@=p�C��                                    Bx�9   �          A(�@�������=�?@  C�f@�������>B�\?��HC��                                    Bx�9+�  �          A(�@�����{>��H@H��C�s3@�����{?\)@c�
C�u�                                    Bx�9:l  �          A(�@�  ��\?!G�@���C���@�  ��\?0��@�C��\                                    Bx�9I  �          A(�@Å��33>�(�@.{C�ff@Å���H>�@B�\C�h�                                    Bx�9W�  �          A  @�(��߮>�{@(�C�j=@�(��߮>Ǯ@\)C�k�                                    Bx�9f^  �          A�R@�z���
=�����(Q�C�ٚ@�z���\)��Q��
=C��R                                    Bx�9u  �          A@�����Q���� Q�C��\@�����׿�  �z�C���                                    Bx�9��  �          A�H@����  ?�@Tz�C�S3@����  ?��@aG�C�T{                                    Bx�9�P  �          Aff@��H��?��HAK�
C�f@��H��?�p�AN�\C��                                    Bx�9��  �          Az�@�(���\)@	��A\z�C��@�(���\)@
�HA^�\C�\                                    Bx�9��  �          AQ�@�\)��  @{Ad(�C�h�@�\)��  @�RAe��C�j=                                    Bx�9�B  �          A  @XQ����R@�Az�RC��H@XQ����R@(�A{�C���                                    Bx�9��  �          A�
@Z=q�   @�RAfffC���@Z=q�   @�RAf�\C���                                    Bx�9ێ  �          A�@����Q�@hQ�A���C���@���ȣ�@hQ�A�\)C���                                    Bx�9�4  �          A�@��\�أ�@ffAt  C��@��\�أ�@As
=C�)                                    Bx�9��  �          A�@������@   A��HC���@�����@�RA�  C��                                    Bx�:�  �          A33@����  @"�\A�(�C�W
@����  @!G�A���C�S3                                    Bx�:&  �          Aff@ҏ\��p���33�\)C�f@ҏ\�����p����C��                                    Bx�:$�  �          A�H@������ý#�
��z�C�g�@������ý��
���HC�g�                                    Bx�:3r  �          A�\@�����ff>��?p��C�=q@�����ff=�G�?0��C�=q                                    Bx�:B  �          A33@�p������33��RC��@�p�����Ǯ��RC��                                    Bx�:P�  �          Ap�@�{��(�>aG�?�{C���@�{��z�>.{?���C���                                    Bx�:_d  �          A��@�{���
=L��>��
C���@�{���
�#�
�L��C���                                    Bx�:n
  �          Ap�@�����{����
=C���@�������{��C���                                    Bx�:|�  �          Aff@����zΎ��33C�l�@����(���z��C�t{                                    Bx�:�V  �          AQ�@�G���녿k����
C��R@�G������z�H�ȣ�C��q                                    Bx�:��  |          A33@�z���33��\)��
=C��@�z���33���E�C��                                    Bx�:��  �          A  @߮��  ���33C�C�@߮������R�{C�O\                                    Bx�:�H  �          A=q@�G���Q��\�3�C�+�@�G��߮��{�<��C�7
                                    Bx�:��  �          A�@�p����ÿ�
=�+33C���@�p���Q��\�4��C�ٚ                                    Bx�:Ԕ  �          A�@��H��33��\)�p�C�T{@��H���\�����C�aH                                    Bx�:�:  �          Aff@�=q��  @�\At��C��@�=q��G�@��Aj�RC��)                                    Bx�:��  �          Aff@ڏ\��  ?��H@�(�C���@ڏ\����?���@��C��=                                    Bx�; �  �          A33@����ff@
=qA`(�C�\)@����\)@�
AU��C�Ff                                    Bx�;,  �          A�
@�p�����?��
AC���@�p�����?�
=A33C�|)                                    Bx�;�  �          A�@أ���@&ffA��C�K�@أ���
=@   A��\C�+�                                    Bx�;,x  �          A��@�=q��G�@z�AT��C�N@�=q���\?��HAIG�C�5�                                    Bx�;;  �          A  @�z����?�33A,(�C���@�z���{?��A�
C��f                                    Bx�;I�  �          A�H@�
=���?�
=A.=qC�@�
=����?���A"{C��R                                    Bx�;Xj  �          A�@θR��?�{A?
=C���@θR��
=?�p�A1��C��H                                    Bx�;g  �          A�@��H��
=?�ffA   C���@��H��  ?�Q�A�
C��)                                    Bx�;u�  |          A�R@�G����H?Tz�@�z�C�S3@�G����?:�H@�ffC�Ff                                    Bx�;�\  �          Az�@�p���{�Tz���C��@�p���p��s33��(�C��)                                    Bx�;�  �          A��A (�����xQ���Q�C�Y�A (���
=��=q��ffC�k�                                    Bx�;��  �          AG�@�ff��
=��p��33C���@�ff����{��
C���                                    Bx�;�N  �          A�@�  ��z�>�{@	��C�ٚ@�  ����>8Q�?��C���                                    Bx�;��  �          A�
@ٙ���33?��A%��C��@ٙ���z�?�p�Ap�C���                                    Bx�;͚  �          A(�@ָR���
?�=qAC���@ָR���?�z�Az�C���                                    Bx�;�@  �          A�@˅��z�?��A
=C�n@˅��p�?�33@���C�Y�                                    Bx�;��  T          A��@�����=q�#�
���\C�s3@�����녾��R��
=C�xR                                    Bx�;��  �          A�@�����    ���
C���@���������p��C��=                                    Bx�<2  �          A��@�����?\)@\(�C��@�����{>Ǯ@��C�{                                    Bx�<�  �          A33@��
���
@
=AT��C�~�@��
��?�Q�AC\)C�T{                                    Bx�<%~  �          A��A  ����?B�\@��C�=qA  ��=q?�R@s33C�.                                    Bx�<4$  �          Ap�A=q��Q�=u>�Q�C�8RA=q��Q콣�
���C�8R                                    Bx�<B�  �          AG�A����Q�>�\)?��HC�g�A������>�?Q�C�b�                                    Bx�<Qp  �          Ap�AQ�����?&ff@~�RC�Y�AQ���G�?   @C�
C�K�                                    Bx�<`  �          A��A(����H?�=q@�ffC��{A(���(�?p��@���C���                                    Bx�<n�  �          A��A{���H?�=q@ָRC���A{��(�?n{@�\)C��)                                    Bx�<}b  �          Az�@�����H?�
=@�\C��@����(�?�  @�ffC�                                      Bx�<�  �          A\)@�\���R?���@�C���@�\���?\(�@�z�C��{                                    Bx�<��  �          A  @�����?���A�C�"�@����G�?��
A Q�C���                                    Bx�<�T  �          A\)@��\���
@�AP��C�j=@��\��ff?�z�A?33C�1�                                    Bx�<��  �          A�H@�33���@G�AeG�C��@�33��  @ffAS�
C�˅                                    Bx�<Ơ  �          A
=A   ���R?У�A#�C��A   ����?��HA�C��                                    Bx�<�F  �          Aff@��H��=q?}p�@ǮC�޸@��H���?L��@�G�C��                                    Bx�<��  �          A�@�����>�ff@2�\C��@����(�>u?�  C�H                                    Bx�<�  �          A\)@�=q��ff?�G�@�=qC�� @�=q���?O\)@�G�C�e                                    Bx�=8  �          A33@�33��Q�?���@�G�C�K�@�33���?z�H@���C�+�                                    Bx�=�  �          A�@�p�����?��
A (�C�"�@�p���=q?�=q@׮C��q                                    Bx�=�  T          A�
@�����?z�H@�33C�s3@������?E�@�Q�C�XR                                    Bx�=-*  �          A(�@������R?.{@�
=C��q@�����\)>�ff@333C���                                    Bx�=;�  �          A\)@�Q����?+�@��C���@�Q���>�G�@.{C���                                    Bx�=Jv  �          A�A���z�?
=q@W�C��RA����>���@�
C��=                                    Bx�=Y  �          AQ�A z���33�#�
�aG�C�
A z���33����˅C�)                                    Bx�=g�  �          A\)@�p�����>W
=?���C�'�@�p����׼#�
��\)C�#�                                    Bx�=vh  �          A33@�ff���@4z�A��C�s3@�ff��(�@%A�C�3                                    Bx�=�  �          A�@�33���@@  A�=qC��@�33���R@0  A��C���                                    Bx�=��  �          A�H@�\���@333A�(�C���@�\��(�@#33A�G�C�1�                                    Bx�=�Z  �          A�R@陚��(�@AG�A�Q�C�Ф@陚����@1�A�p�C�ff                                    Bx�=�   T          A
=@�33����@U�A�{C���@�33��=q@FffA�p�C��                                    Bx�=��  �          A�\@������@b�\A��C���@������@Tz�A��C��                                    Bx�=�L  T          A=q@������@L��A��C�*=@������@<��A��RC���                                    Bx�=��  �          Aff@���{@EA�ffC�u�@���33@6ffA�G�C�                                      Bx�=�  �          AG�@���(�@dz�A�G�C�#�@���=q@U�A�ffC��3                                    Bx�=�>  �          A�@�ff�tz�@|��A��HC�@�ff��G�@n�RA\C�\)                                    Bx�>�  �          A��@�\�e�@y��A�ffC��q@�\�s33@l(�A���C�P�                                    Bx�>�  �          A(�@��H�_\)@tz�A�\)C�G�@��H�mp�@g
=A�C��)                                    Bx�>&0  
�          AQ�@����P��@o\)A�C�O\@����^{@b�\A���C���                                    Bx�>4�  �          A�@��P  @l(�A¸RC�C�@��^{@`  A��C���                                    Bx�>C|  �          AQ�@�Q��h��@UA�z�C��@�Q��u�@G�A�Q�C�~�                                    Bx�>R"  �          A�
@�33�l��@AG�A��C��@�33�w�@333A��C��H                                    Bx�>`�  �          A\)@����k�@U�A�\)C��@����xQ�@EA��\C�+�                                    Bx�>on  �          A�\@�
=���
@"�\A�z�C���@�
=����@�AmG�C�"�                                    Bx�>~  �          A�@��R�O\)@.�RA��HC��
@��R�Z=q@!�A��C��                                    Bx�>��  T          AG�@�����
=@  Al��C�+�@�����33?�p�AO�C�Ǯ                                    Bx�>�`  T          Ap�@�Q��\)@z�AtQ�C�f@�Q���(�@�
AXz�C��)                                    Bx�>�  �          A��@�  ��Q�@p�Aip�C��\@�  ��z�?���AMG�C��=                                    Bx�>��  �          A�@���hQ�@{A�  C��@���q�@�RAmp�C��f                                    Bx�>�R  �          A
�H@�G��_\)@(�A���C��
@�G��h��@p�Al(�C�q                                    Bx�>��  �          A
=@���Tz�@+�A��
C�*=@���_\)@��A�\)C��                                     Bx�>�  T          A
�\@�  �]p�@!G�A�C��q@�  �g�@�Au�C�)                                    Bx�>�D  �          A��@����U@33As33C�e@����^�R@z�AZ{C��\                                    Bx�?�  �          A(�@�z��l(�@
=A_�
C�R@�z��u�?�{AD  C��\                                    Bx�?�  T          A33@����G�@�\AYC��R@����?޸RA8��C�W
                                    Bx�?6  �          AQ�@�\�z=q?�33AIp�C�]q@�\����?У�A+�C���                                    Bx�?-�  �          A��@�R��\)?��AG
=C�8R@�R���?˅A&�RC��q                                    Bx�?<�  �          Az�@�\��z�?��RAffC��3@�\��
=?h��@�{C��)                                    Bx�?K(  �          Ap�@�  ���\?��\@�z�C�N@�  ��z�?0��@�\)C�!H                                    Bx�?Y�  �          Az�@�����?��HAO�
C��\@�����?�33A-�C��                                    Bx�?ht  �          A��@��H�J=q@{A�=qC�
=@��H�U�@�RAj�HC��H                                    Bx�?w  �          Az�@������@
=Az=qC���@�����R@�
AY�C�4{                                    Bx�?��  �          A��@����G�?��
A;33C�{@�����?�Q�A�RC���                                    Bx�?�f  �          A��@�33����?�@��
C�W
@�33���?G�@��
C�%                                    Bx�?�  �          Ap�@陚���?��AQ�C�N@陚��z�?p��@ÅC��                                    Bx�?��  D          A@�
=��{>�@@��C���@�
=���R=L��>���C���                                    Bx�?�X  �          A\)@�=q�N{@��
BffC��@�=q�e�@��B�RC��R                                    Bx�?��  �          A
=q@�z��l(�@��HA�ffC�w
@�z���  @r�\A�z�C�xR                                    Bx�?ݤ  �          A\)@�33��@4z�A��C�@�33��z�@�RA�z�C�%                                    Bx�?�J  �          A
{@�Q���
=@=qA�{C���@�Q�����@G�AZ{C��                                    Bx�?��  �          A��@����\@j�HA�C��@����@P  A���C�S3                                    Bx�@	�  �          A
�H@ʏ\���?�Q�AO�C�\)@ʏ\��{?��RA�RC��                                    Bx�@<  T          A	@Å���R@��A�\)C��@Å��z�@   AX��C���                                    Bx�@&�  �          A
{@�{��\)?���AEC�8R@�{�˅?�=qA=qC���                                    Bx�@5�  �          A	�@�ff�˅?�G�A>=qC�L�@�ff�Ϯ?��RA��C��                                    Bx�@D.  �          A
�R@�����ff@�RA�p�C�Ff@�����z�@   AV=qC�Ф                                    Bx�@R�  �          A	@����\)@��A��\C�|)@�����?�
=AN�HC�3                                    Bx�@az  �          A
�H@��
���@&ffA�  C��@��
��Q�@
=AaG�C�*=                                    Bx�@p   �          A
ff@����
=@�\AZ{C��{@�����
?�p�A�C���                                    Bx�@~�  �          A  @�
=���@QG�A��C���@�
=��=q@3�
A��HC��                                    Bx�@�l  �          A��@�\)���@e�A�\)C��)@�\)��ff@G�A��\C�5�                                    Bx�@�  �          A(�@�(���
=@0  A���C���@�(���{@��Aw\)C���                                    Bx�@��  �          A��@�Q���  @!�A��C��@�Q���ff@�\A]G�C�B�                                    Bx�@�^  �          A(�@�\)��p�@(��A��HC��=@�\)��(�@	��Ak33C�^�                                    Bx�@�  �          Az�@�����@z�Aap�C�(�@������?�=qA+
=C��                                     Bx�@֪  �          A	��@�{��=q@�
A]��C�7
@�{��\)?�=qA)�C��=                                    Bx�@�P  T          A	��@�����?�Q�A{C���@����
?p��@ȣ�C�l�                                    Bx�@��  �          A��@�
=�ə�?�ff@���C�Ф@�
=���
>��@R�\C���                                    Bx�A�  �          @���@�p���ff?xQ�@�33C�33@�p���Q�>\@333C��                                    Bx�AB  �          @�p�@θR�L(�@p�A�ffC��R@θR�XQ�?�z�Aj�RC�.                                    Bx�A�  �          @��@�G�@\)@��
B
G�A��
@�G�?���@��B�\A���                                    Bx�A.�  �          @��H@�33@5�@�B��A�
=@�33@
=@�p�B%�HA���                                    Bx�A=4  �          @��@��@(��@�G�B((�A�  @��@��@�Q�B1  A��                                    Bx�AK�  �          @��R@�z�@z�@�p�B{A���@�z�?���@��HBp�As\)                                    Bx�AZ�  �          A z�@���?�(�@�  BQ�A3�
@���?@  @��HB@��                                    Bx�Ai&  �          A{@�p�?�Q�@~�RA��A;�@�p�?�ff@�33A���A
=q                                    Bx�Aw�  �          A z�@�G��(��@~�RA��C��@�G����@y��A��C�|)                                    Bx�A�r  �          @�z�@ʏ\�33@�33A��HC�@ʏ\�+�@w
=A���C��f                                    Bx�A�  T          @�33@��H��@j=qA��C���@��H���@]p�A�G�C���                                    Bx�A��  �          @�=q@��H���@7
=A�(�C�` @��H�)��@'
=A��RC�e                                    Bx�A�d  �          @�=q@�33����@��\B �RC�)@�33���H@{�A���C�p�                                    Bx�A�
  �          @���@�{�L��@=p�A��HC�˅@�{�^{@(Q�A�=qC���                                    Bx�Aϰ  �          @�=q@ʏ\�   @p  A��HC�9�@ʏ\�7
=@_\)A�G�C���                                    Bx�A�V  �          @�=q@��
��=q?�Adz�C��3@��
��G�?�  APQ�C�Ff                                    Bx�A��  �          @��@�\��
@Au�C�(�@�\���?��A\(�C�q�                                    Bx�A��  �          @�{@�  �{@z�A�{C�}q@�  �(�@At(�C���                                    Bx�B
H  �          @�{@�z��33?��RAhQ�C�:�@�z��\)?�\AN�\C���                                    Bx�B�  �          @�(�@��
�H@(�A�G�C���@��Q�?��HAf�RC��=                                    Bx�B'�  �          @�G�@�׿�Q�?�AH��C���@���ff?�(�A/�C��                                    Bx�B6:  �          @�Q�@�33���\?.{@��C�AH@�33��=q?��@�=qC�f                                    Bx�BD�            @�Q�@���p�>�{@�RC�z�@���G�>8Q�?��C�aH                                    Bx�BS�  �          @�{@���
=?aG�@�=qC��R@���G�?:�H@�C�J=                                    Bx�Bb,  �          @���@�G���
?#�
@��C��@�G��
=>�
=@J�HC��                                    Bx�Bp�  �          @�(�@陚��G�?���A   C�7
@陚���?���A��C��q                                    Bx�Bx  �          @�\)@�Q쿕?�{A"ffC��3@�Q쿧�?�p�A�RC��                                    Bx�B�  �          @���@����H>Ǯ@8��C��3@��޸R>W
=?��
C�u�                                    Bx�B��  �          @��@��Ϳfff>�@c�
C��@��Ϳs33>\@5�C�y�                                    Bx�B�j  T          @�
=@�ff��\)�u��G�C���@�ff��\)��Q�(��C���                                    Bx�B�  T          @���@�{�O\)�L�����C�  @�{�8Q�aG��У�C�U�                                    Bx�Bȶ  �          @�z�@��\�!G�?   @h��C��3@��\�.{>�(�@G�C��                                    Bx�B�\  �          @��H@�ff��G�?h��@�C�>�@�ff��{?J=q@��HC���                                    Bx�B�            @�  @��H����>�(�@J=qC��q@��H��z�>�=q@�C���                                    Bx�B��  2          @��@�ff����?u@���C�#�@�ff��z�?B�\@�(�C���                                    Bx�CN  �          @�@�(���(�@��A�ffC�1�@�(��ٙ�@�Aw
=C�N                                    Bx�C�  �          @�33@�
=��p�@q�A��C�� @�
=���@g�A�C��                                    Bx�C �  �          @�=q@��Ϳz�@|(�A�{C�e@��Ϳ��\@vffA��
C�~�                                    Bx�C/@  �          @��H@�=q�8Q�@���B��C�"�@�=q�.{@�\)B  C��=                                    Bx�C=�  �          @�@�?�ff@���B	{A��@�?\)@��B\)@��R                                    Bx�CL�  �          @��R@ᙚ>�=q@*�HA��
@p�@ᙚ��@+�A��RC��                                    Bx�C[2  �          @��@�(�?�{@Tz�A�  A
=@�(�?:�H@[�A��H@��H                                    Bx�Ci�  T          @�ff@�Q�@   @�Q�A�33A��@�Q�?��
@��RB��A\Q�                                    Bx�Cx~  �          @�@���@z�@�B!�A��@���?��R@�(�B*
=Aw�                                    Bx�C�$  �          @�  @��?Tz�?@  A�
A�
@��?=p�?W
=A�
A                                      Bx�C��  �          @��@����p��<(���z�C�s3@���#�
�>{��  C�ٚ                                    Bx�C�p  T          @��@��H�!G��.{��p�C�u�@��H���R�1G�����C���                                    Bx�C�  �          @�{@����(���ff�Z{C�S3@���aG����^�HC�!H                                    Bx�C��  �          @�z�@�?J=q>�\)@�@�
=@�?@  >�p�@2�\@�{                                    Bx�C�b  �          @��H@�G�@��?�{A&�RA���@�G�@�?���AEG�A�ff                                    Bx�C�  �          @��H@�=q@=q>���@J�HA�p�@�=q@?.{@��A��                                    Bx�C��  �          @�33@�{@��?n{@ᙚA�Q�@�{@�?�
=A\)A�
=                                    Bx�C�T  �          @�  @�>�(�����{@X��@�>��H���p��@w
=                                    Bx�D
�  	�          @�  @�>\)��  �?�{@�>�z῜(���\@\)                                    Bx�D�  "          @�@�\�W
=��  ��{C��
@�\�8Q쿋���\C�1�                                    Bx�D(F  "          @�Q�@��H��\)��p��5C���@��H���Ϳ�  �8��C���                                    Bx�D6�  
�          @�@�33��(���(���(�C�7
@�33�B�\� ������C�4{                                    Bx�DE�  
�          @�@�\)?�\)�"�\��{Ai��@�\)?�z��z����A��\                                    Bx�DT8  
�          @��H@љ�?�(���=q�t(�AI�@љ�?�����Z�\Aep�                                    Bx�Db�  "          @���@�33>B�\�u�	�R?���@�33?&ff�r�\��@��
                                    Bx�Dq�  �          @��@��\?����������AB�H@��\?�
=�vff��HA�{                                    Bx�D�*  �          @�Q�@�G���zῢ�\��
C�j=@�G���  ��z����C�o\                                    Bx�D��  �          @�\@�33���?=p�@���C�/\@�33���>��?��HC�f                                    Bx�D�v  �          @�G�@�R�5��ff�a�C�+�@�R�(�ÿ���{C�g�                                    Bx�D�  �          @�G�@��0�׽�G��^�RC�H�@��+��B�\�\C�\)                                    Bx�D��  �          @�(�@�\)?(��?��A�@�{@�\)?�?�{A
=@���                                    Bx�D�h  �          @�  @��
��  ?�  AI�C���@��
��?��A2�RC���                                    Bx�D�  "          @�p�@������
>�p�@6ffC�<)@�����(���z����C�8R                                    Bx�D�  �          @�\)@S33���ͿG���33C���@S33��  �У��C�C��                                    Bx�D�Z  
�          @��R@�ff�����8Q쿯\)C��@�ff��\)�n{��
=C��                                    Bx�E   �          @���@�p����þ�G��S�
C�7
@�p�����33��C�u�                                    Bx�E�  
�          @�z�@��
��=q?���A#�C��q@��
��{?z�@�z�C���                                    Bx�E!L  �          @��H@����ff?�33A(�C�5�@������>�(�@Q�C��
                                    Bx�E/�             @��
@��
����?s33@�{C�
@��
���H>aG�?�
=C���                                    Bx�E>�  �          @���@�(����?�\@uC��3@�(���=q��  ���C���                                    Bx�EM>  
�          @��
@�\)��  >\@7�C�q@�\)��Q쾏\)���C�R                                    Bx�E[�  T          @���@����
==���?5C���@����{�!G���
=C��3                                    Bx�Ej�  
�          @��@��
��>���@<(�C�b�@��
���Ǯ�8��C�b�                                    Bx�Ey0  �          @���@����ȣ�>�G�@N{C��R@������þ�Q��,(�C���                                    Bx�E��  2          @�@��
�Ǯ?s33@���C���@��
���>��?�z�C��\                                    Bx�E�|  
J          @��R@�33��p����R�ffC���@�33���\�����   C��                                    Bx�E�"  �          @�p�@�  ��{�{��\)C��
@�  ���\�G
=��
=C��q                                    Bx�E��  �          @�z�@�ff�����C33��G�C�� @�ff���H�g
=���HC��                                    Bx�E�n  
Z          @�(�@�=q��p��{��33C�Y�@�=q��33�333��p�C�]q                                    Bx�E�  T          @�@����g�����  C��{@����dz�������C��q                                    Bx�Eߺ  T          @���@��
��p�����_�C�P�@��
�y���z����C�33                                    Bx�E�`  "          @���@����p�@z�A���C���@������?���A8(�C�>�                                    Bx�E�  �          @��@�Q���@<(�A��HC���@�Q�����@�A��RC��H                                    Bx�F�  T          @�p�@��R�l(�@l(�A�ffC�S3@��R��z�@K�A�ffC���                                    Bx�FR  T          @�p�@�Q���
=?�G�A_33C�*=@�Q����?��AC���                                    Bx�F(�  �          @�ff@�G����\@h��A�\)C���@�G�����@>�RA��C��{                                    Bx�F7�  T          @�=q@|������@,��A���C�1�@|����=q?�Am��C���                                    Bx�FFD  
�          @���@}p���ff?��A!C�G�@}p���=q>�@n{C��                                    Bx�FT�  "          @�@�����z�?
=@��
C��3@�������aG����HC��                                    Bx�Fc�  �          @�
=@�(���Q�>�\)@��C��@�(�������\)C��                                    Bx�Fr6  "          @��@w
=��ff?��Az�C���@w
=����>k�?�C��                                    Bx�F��  
�          @��H@AG�����?��HA1p�C��@AG���p�?
=q@��
C��                                    Bx�F��  
�          @�p�@J=q��ff?�AhQ�C��q@J=q����?�G�@��HC�L�                                    Bx�F�(  "          @�
=@;���G�@G�A�=qC���@;�����?��A�C�O\                                    Bx�F��  
�          @�z�@)�����@-p�A�
=C�|)@)����R?�\AO33C��                                    Bx�F�t  "          @��@6ff��Q�@P  A�  C�@ @6ff��z�@Q�A�C��H                                    Bx�F�  
�          @�p�@Fff���@{A�Q�C��@Fff�Å?�A;33C�s3                                    Bx�F��  D          @�ff@C�
��  �8Q��  C�}q@C�
���z�H�C��                                    Bx�F�f             @�\)@+���
=@,��A�\)C��=@+���G�?�z�A�
=C��
                                    Bx�F�  �          @�  @�R��z�?�\)A��RC�g�@�R���H?��A�HC��                                    Bx�G�  �          @��H@Q����@z�A��C�(�@Q����H?��A.ffC��=                                    Bx�GX  �          @أ�?����?�Q�AT��C��)?����=q?�R@�
=C�u�                                    Bx�G!�  T          @�(�@�z�����?�G�AP��C��=@�z����R?O\)@�\)C�:�                                    Bx�G0�  D          @ڏ\@�����@@��AӮC�� @�����@=qA���C��                                    Bx�G?J             @ָR@6ff���@K�A�  C��{@6ff��  @��A��HC��
                                    Bx�GM�  
�          @�ff@�
=��\)@Y��A�33C�t{@�
=����@2�\A�C�
                                    Bx�G\�  "          @�ff@�  �o\)@�B=qC�%@�  ����@hQ�A�RC�=q                                    Bx�Gk<  T          @���@�(��4z�@�(�B,��C��q@�(��]p�@�ffBQ�C��
                                    Bx�Gy�  �          @�  @��\��@��BN��C�ٚ@��\�&ff@�33B?�\C��                                     Bx�G��  
�          @�G�@_\)�.{@�Q�Bo\)C��R@_\)�}p�@�Bj33C��                                    Bx�G�.  �          @ۅ@�ff�(�@���B%\)C��=@�ff�@��@y��B(�C�4{                                    Bx�G��  �          @�\)@r�\��@0��A�z�C�H�@r�\����@�A���C�@                                     Bx�G�z  �          @�=q?������l����RC�%?������
��33�*=qC�O\                                    Bx�G�   v          @�(�@�����33@�RA�p�C�aH@������
?��ATz�C��                                     Bx�G��             @���@�����33@ffA���C��\@�����33?�p�A<Q�C�Ǯ                                    Bx�G�l  
�          @��@�\)���@7
=A��C��@�\)���@�A�
=C���                                    Bx�G�  �          @�ff@�z�����@(��A�=qC�"�@�z���33@G�A��\C�R                                    Bx�G��  �          @�(�@�p��l��@J=qA�p�C�0�@�p����@'�A��HC��                                    Bx�H^  
�          @�p�@��>��@�G�B-G�@Fff@�녾�p�@���B,�HC�Ǯ                                    Bx�H  T          @ᙚ@�����@��B{C�p�@����33@��
BG�C��H                                    Bx�H)�  "          @�(�@�(��!G�@�33B!p�C���@�(����\@�
=B�
C�aH                                    Bx�H8P  
�          @ᙚ@�녿��R@��
B�HC�j=@�녿�@���B�C�p�                                    Bx�HF�  �          @��@���� ��@uB�C�H@����!�@aG�A��\C��
                                    Bx�HU�  
�          @�\@�ff���
@z=qB33C��\@�ff��@l(�A��C�/\                                    Bx�HdB  �          @�ff@�G��7�@g�A�33C�Q�@�G��Vff@L(�AѮC�n                                    Bx�Hr�  
�          @�33@�\)�3�
@aG�A�(�C�p�@�\)�QG�@EA��HC��{                                    Bx�H��  �          @��H@����n�R@0��A���C�n@������\@p�A�=qC�33                                    Bx�H�4  "          @�(�@���p  @aG�A�33C�N@�����R@<��A�ffC���                                    Bx�H��  T          @��@�z��n{@B�\A��HC��@�z����@\)A���C��=                                    Bx�H��  D          @޸R@��\�c�
@]p�A�C�Ǯ@��\����@:�HAǮC��                                    Bx�H�&  
          @�p�@����aG�@vffB��C�{@�������@Tz�A�C�,�                                    Bx�H��  "          @޸R@��\�j=q@mp�B=qC���@��\����@I��A�Q�C���                                    Bx�H�r  "          @�
=@��
�Fff@c33A�p�C�ff@��
�dz�@E�A��
C���                                    Bx�H�  "          @���@��׿�p�@�HA���C��@��׿�@(�A���C��H                                    Bx�H��  �          @�  @�
=�L��@K�A�ffC���@�
=�g�@,��A��C���                                    Bx�Id  �          @��@�{�Fff@Z�HA��HC��@�{�c�
@<(�A�ffC�)                                    Bx�I
  �          @�G�@��H����@n{B�C��3@��H��@]p�A��C�\)                                   Bx�I"�  �          @��
@�33�q�@<(�Ȁ\C��@�33���@Q�A�33C��3                                   Bx�I1V  �          @�  @����|��@<��A���C�q�@������\@ffA���C�#�                                    Bx�I?�  �          @ٙ�@�p���
=?�z�A�C�!H@�p���ff?�G�A+�C�k�                                    Bx�IN�  �          @أ�@mp����\>\@P��C��3@mp���=q��
=�dz�C��{                                    Bx�I]H  �          @�=q@W����H����~�RC���@W���
=��ff�1�C�AH                                    Bx�Ik�  �          @�z�@AG��������p�C�>�@AG���=q�G���33C���                                    Bx�Iz�  �          @��@J=q��(��h����\)C�!H@J=q��{�޸R�o
=C��f                                    Bx�I�:  �          @�G�@p  ��(��+���\)C�@ @p  ��\)��  �F=qC���                                    Bx�I��  �          @��
@S�
��
=�����\C��H@S�
���
�7
=���C�AH                                    Bx�I��  �          @��
@E��Q쾀  �Q�C���@E��p���{�
=C�˅                                    Bx�I�,  �          @߮@P  ����?�{A�\C�"�@P  ��z�>k�?��C���                                    Bx�I��  �          @�@)����=q�L�;�ffC�Y�@)����  �u� ��C�w
                                    Bx�I�x  �          @�G�@S�
��{@333A�Q�C��
@S�
����@ffA��C���                                    Bx�I�  �          @�G�@Fff��(�?�A�(�C���@Fff���H?�ffAp�C�~�                                    Bx�I��  �          @�G�@\)��=q@��Bz�C���@\)���
@`  Bz�C���                                    Bx�I�j  �          @�p�@�ff���R?�(�A)�C�q@�ff���\>��H@���C��                                    Bx�J  �          @ҏ\@����)��@��
B(�HC�J=@����P  @|(�BQ�C���                                    Bx�J�  �          @��
@������@��A�G�C�xR@�����=q?�  AW�C���                                    Bx�J*\  �          @�33@fff��  ��z��#33C���@fff���Ϳ�=q�\)C��3                                    Bx�J9  �          @�33@�Q���=q>L��?�(�C��
@�Q����������RC��                                    Bx�JG�  �          @ᙚ@�\)��
=�A����C��{@�\)�~�R�h��� ffC�XR                                    Bx�JVN  �          @�(�@tz������g���\)C�
@tz��}p�����Q�C�H                                    Bx�Jd�  �          @�\@c33�����@����=qC��@c33����p  � =qC�9�                                    Bx�Js�  �          @�\@C33����(����p�C��
@C33���
�\����C�˅                                    Bx�J�@  �          @�ff@(����H�=q��\)C�� @(����P���ᙚC���                                    Bx�J��  �          @�@XQ���{�\)���
C��@XQ������P  �ᙚC���                                    Bx�J��  �          @ڏ\@Dz����
����RC�T{@Dz���
=�HQ���z�C�:�                                    Bx�J�2  �          @���@W
=����33��\)C��)@W
=��(��4z���p�C��R                                    Bx�J��  �          @ڏ\@I�����R�33���C�l�@I������7
=�ŮC�8R                                    Bx�J�~  �          @��
@#33��녿�z����C�g�@#33��
=�0����z�C�H                                    Bx�J�$  T          @ڏ\@z���(���\)�~ffC�` @z������.�R���C��=                                    Bx�J��  �          @�G�?�
=���ÿ��xz�C�}q?�
=���R�,����Q�C��H                                    Bx�J�p  �          @��@�\)��\)@(Q�A�G�C�*=@�\)����?��Aw�
C�E                                    Bx�K  �          @��@��R���\@>�RA�p�C���@��R���R@  A�z�C�z�                                    Bx�K�  �          @��@�����@33A��C��\@�����
=?��AXz�C�
=                                    Bx�K#b  �          @߮@�����33?�p�A��
C�R@������H?�{A4z�C�J=                                    Bx�K2  �          @ᙚ@�p���33?�{Au�C���@�p����\?��HA�HC�=q                                    Bx�K@�  �          @�
=@�(����?ٙ�Ab�\C��@�(���  ?h��@�C�q�                                    Bx�KOT  �          @���@�ff��(�?Tz�@�\)C�}q@�ff��{=��
?.{C�O\                                    Bx�K]�  �          @޸R@��H��z�?p��@���C��3@��H��\)>��@
�HC��                                    Bx�Kl�  �          @Ӆ@����e�����ffC�:�@����^{�����
C���                                    Bx�K{F  �          @�  @��H��\)?��
A/33C�#�@��H���
?(��@��\C��                                    Bx�K��  �          @��
@�����G�?^�R@�\C�k�@������
>u@�\C�'�                                    Bx�K��  �          @ۅ@��r�\?(�@�z�C�#�@��u=u?   C��R                                    Bx�K�8  T          @��@����  >u@��C��@���\)��{�=p�C�                                      Bx�K��  �          @�p�@�Q��Z�H?��AV{C�3@�Q��g
=?��
A�\C�b�                                    Bx�KĄ  �          @�Q�@���p  ?��ARffC���@���{�?z�HAG�C���                                    Bx�K�*  �          @ۅ@��
����?�\)A8��C��R@��
��p�?E�@ϮC��                                    Bx�K��  �          @��@��\�r�\?���AB�RC��@��\�}p�?c�
@�p�C�K�                                    Bx�K�v  T          @���@��H�Dz�@�A�C�(�@��H�XQ�?��HA���C��
                                    Bx�K�  �          @�@�  �Q�@��A��C��
@�  �,��@�A��HC�S3                                    Bx�L�  1          @�G�@�(��
=?�z�A~{C��3@�(���?˅AR{C��\                                    Bx�Lh  �          @�@��ÿ�=q@�RA���C���@����Q�?�Q�A���C���                                    Bx�L+  �          @߮@�z῁G�?�A�C���@�zῢ�\?�G�Ah��C��3                                    Bx�L9�  �          @���@�=q�8Q�?�(�AH��C���@�=q�n{?���A8  C���                                    Bx�LHZ  �          @��H@�p��Ǯ?��A;�C�P�@�p��
=?���A1�C�z�                                    Bx�LW   �          @���@�
=��Q�?�{A7
=C���@�
=��\)?�=qA3\)C��                                    Bx�Le�  �          @�@أ׾\)?��
A+�C�e@أ׾���?�  A'33C���                                    Bx�LtL  �          @�  @�ff���
?=p�@�=qC���@�ff�8Q�?8Q�@��C�=q                                    Bx�L��  "          @�@�녿��?�\@��C�=q@�녿���>�{@5�C��)                                    Bx�L��  �          @޸R@����\��\)�z�C�� @���   �\)��
=C��                                    Bx�L�>  
�          @���@��
�   ���
�*=qC��\@��
��ff�Ǯ�N{C�%                                    Bx�L��  �          @��H@�녾aG���33�<(�C�
@�녾#�
�\�J�HC�O\                                    Bx�L��  
�          @�@��;�Q�����U�C��f@��;�z��ff�n{C�Ǯ                                    Bx�L�0  "          @ۅ@�녿
=�W
=���
C��@�녿\)��z��{C���                                    Bx�L��  T          @�(�@�  ��ff���aG�C���@�  ���
�W
=��\C��q                                    Bx�L�|  T          @أ�@�33��(�=L��>�C���@�33��(��#�
��=qC��q                                    Bx�L�"  T          @��
@׮��G����
�#�
C���@׮��  �L�Ϳ�C�                                    Bx�M�  �          @�  @Ӆ���\=u?   C���@Ӆ��G�����ffC���                                    Bx�Mn  �          @�z�@�ff��  ��\)��RC�q�@�ff��p���z��{C��                                    Bx�M$  �          @�p�@�(�����ff�xQ�C��@�(���p��333�\C�h�                                    Bx�M2�  �          @�z�@�\)�  �.{��(�C��@�\)��;��H���C�B�                                    Bx�MA`  
�          @���@�����ÿ�R���\C�!H@�����H�Tz���\)C���                                    Bx�MP  
�          @׮@��
��>.{?�p�C�&f@��
��\�����\C�"�                                    Bx�M^�  T          @أ�@���,(�?��RA)��C��@���5?W
=@�{C�s3                                    Bx�MmR  "          @�G�@���/\)?(��@�=qC��@���333>��@p�C�Ǯ                                    Bx�M{�  �          @�  @���(Q�>W
=?�\C�t{@���(Q�.{��Q�C�q�                                    Bx�M��  �          @ٙ�@ƸR���?=p�@��HC�9�@ƸR�!�>\@R�\C��                                    Bx�M�D  "          @�G�@���Q녾u��\C�B�@���N{�333����C���                                    Bx�M��  �          @��
@�Q���(�?��@���C�"�@�Q�����\)���HC��                                    Bx�M��  "          @�\)@�p����?=p�@��
C��H@�p����
=�?�G�C��                                    Bx�M�6  �          @��@��R��  ?^�R@�(�C��@��R���\>��@ffC��\                                    Bx�M��  �          @�Q�@�  ���?���A��C��
@�  ��
=>��@���C���                                    Bx�M�  �          @�Q�@���|(�?W
=@�C���@������>�\)@
=C��\                                    Bx�M�(  
�          @ٙ�@�
=��  >�G�@p  C��=@�
=���׾����C��
                                    Bx�M��  T          @�z�@�ff���
?0��@���C�s3@�ff��=�G�?aG�C�B�                                    Bx�Nt  �          @�z�@�33���>���@ ��C��\@�33��녾��R�%�C��\                                    Bx�N  �          @�=q@�����33?Tz�@��C���@�����p�>W
=?�G�C��\                                    Bx�N+�  �          @���@��R����\)�(�C�H�@��R��(��(������C�u�                                    Bx�N:f  "          @�ff@�{�~�R��33�8Q�C��f@�{�x�ÿfff��\)C��{                                    Bx�NI  �          @�=q@��R���H>#�
?���C�q@��R��=q��ff�s�
C�.                                    Bx�NW�  T          @�(�@�G���{?xQ�A�
C��R@�G�����>�{@>{C�L�                                    Bx�NfX  �          @��
@��
��  <�>aG�C��@��
���R�#�
����C�<)                                    Bx�Nt�  "          @׮@����  =�Q�?J=qC��@�����R����33C�Ǯ                                    Bx�N��  �          @�z�@�\)���?�@�G�C�Q�@�\)��z�#�
����C�>�                                    Bx�N�J  T          @ָR@�Q����R��\)��C��f@�Q����Ϳ8Q���ffC���                                    Bx�N��  T          @�=q@ ������@Q�A��C��\@ ������?�ffA9��C�4{                                    Bx�N��  T          @���@�{����>���@333C�=q@�{���þ�{�:=qC�>�                                    Bx�N�<  �          @���@�ff��Q�>�p�@E�C��@�ff��Q쾙���{C��                                    Bx�N��  �          @�(�@�����(�>�Q�@C�
C�,�@�����(����
�)��C�+�                                    Bx�Nۈ  T          @�(�@��\���H<�>aG�C��\@��\�����(����C��                                    Bx�N�.  T          @���@�p���{����Z�HC�%@�p����H�z�H�(�C�xR                                    Bx�N��  T          @ۅ@����c�
��=q��
C�+�@����_\)�B�\��ffC�n                                    Bx�Oz  
�          @��
@������\��Q��>�RC��H@����\)�k���p�C�                                    Bx�O   "          @���@������
�B�\�˅C���@��������B�\���C��)                                    Bx�O$�  �          @���@�����33�.{��z�C�AH@�����
=���\�*{C���                                    Bx�O3l  
�          @ۅ@�\)���;����1�C���@�\)��녿n{���HC�=q                                    Bx�OB  
�          @ڏ\@�����G����ͿO\)C�s3@�������.{����C��                                     Bx�OP�  T          @׮@�����p���=q�z�C�XR@������H�^�R��C���                                    Bx�O_^  T          @�  @�{��G���\��=qC��3@�{����\)�C�                                    Bx�On  
�          @�  @�����
=>�(�@g�C�5�@�����\)�B�\����C�(�                                    Bx�O|�  w          @�
=@�z���G����R�+�C�{@�z���ff�c�
��p�C�Z�                                    Bx�O�P  T          @�z�@�{�����\)��RC�&f@�{��33�+����C�P�                                    Bx�O��  1          @��H@�\)��G�=�G�?xQ�C�3@�\)��Q��\��C�(�                                    Bx�O��  T          @���@������\@p�A���C�#�@������H?���A_�C�N                                    Bx�O�B  �          @�  @����Mp�@L(�A�p�C���@����g�@.�RA�ffC��                                    Bx�O��  T          @�ff@�p��}p�@.{A�\)C��R@�p�����@
=qA�=qC��{                                    Bx�OԎ  
�          @�Q�@������@��A�ffC�xR@�����(�?�
=Ar�\C���                                    Bx�O�4  
�          @Ϯ@���`  ��Q�O\)C�t{@���\�Ϳ\)����C���                                    Bx�O��  T          @�(�@�p��4z�?���A�z�C���@�p��C33?��HAL  C��                                     