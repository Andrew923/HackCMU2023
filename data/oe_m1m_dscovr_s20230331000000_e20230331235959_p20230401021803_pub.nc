CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230331000000_e20230331235959_p20230401021803_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-04-01T02:18:03.927Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-31T00:00:00.000Z   time_coverage_end         2023-03-31T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxsOl@  �          A�@�����\�0�����C�W
@��>Ǯ�@����{@8Q�                                    BxsOz�  T          A33@�R��
=�<(���ffC��@�R>\)�XQ���G�?��                                    BxsO��  !          @�@��H��\������
C�R@��H��
=�J=q�ә�C��                                    BxsO�2  T          @�{@��׿�p���R��ffC��@���=u�8����
=?��                                    BxsO��  �          @�  @��׿�\�   ���C���@��׾8Q��C�
��33C�%                                    BxsO�~  �          @ᙚ@�33��\)�5��ffC��)@�33?���@����  @��                                    BxsO�$  �          @�=q@�ff���7���\)C��H@�ff?�{�,(����\A��                                    BxsO��  �          A�R@�
=����U���ffC���@�
=>.{�tz���
=?�=q                                    BxsO�p  �          A	G�@�\����a���z�C��q@�\>���\)��=q?��R                                    BxsO�  
�          A  @�  ��Q��`  ��z�C��)@�  >L����  ��ff?Ǯ                                    BxsO��  �          A\)@����\����z�C��\@�>L���|(���=q?�G�                                    BxsPb  "          Aff@�׿��P������C�!H@��>.{�o\)�Ӆ?�G�                                    BxsP  �          A33@�
=���\(���ffC���@�
=>B�\�|(���Q�?�Q�                                    BxsP*�  T          A�@�녿�=q�S�
��\)C���@��>B�\�q���Q�?��R                                    BxsP9T  �          A{@�\)��  �P����\)C�0�@�\)>u�l����ff?�{                                   BxsPG�  �          A�R@�G���Q��QG����HC�u�@�G�>�z��j�H��@G�                                   BxsPV�  �          A�@��ͿǮ�O\)��\)C��@���>\�dz���  @<��                                    BxsPeF  �          A��@���(��P�����HC�y�@�>�  �j�H��ff?��H                                    BxsPs�  
�          A\)@���z��N{��
=C���@�>�z��g����@\)                                    BxsP��  �          A�@�׿�\)�Q�����C��=@��>�33�h���Σ�@+�                                    BxsP�8  �          A\)@��H�Y���C33��G�C��@��H?c�
�B�\����@�ff                                    BxsP��  �          A@��׿:�H�?\)���C�N@���?s33�;����@�
=                                    BxsP��  �          A��@�ff�k��G���G�C��
@�ff?\(��HQ���Q�@��
                                    BxsP�*  �          A�@��Ϳh���G���z�C��R@���?^�R�H����G�@�
=                                    BxsP��  �          Az�@��
�^�R�G����C���@��
?fff�G
=����@�\)                                    BxsP�v  T          A�R@�\)�s33�Mp���
=C��H@�\)?aG��N{��(�@�                                      BxsP�  �          A�R@�ff�p���QG�����C�� @�ff?k��Q���G�@���                                    BxsP��  �          AQ�@�녿Tz��P�����
C���@��?�G��N{���H@��H                                    BxsQh  "          A��@�\�8Q��Q���Q�C�N@�\?�{�J�H���A��                                    BxsQ  �          AG�@�\�Tz��U����HC���@�\?�ff�QG����@��
                                    BxsQ#�  "          A�@�R�B�\�X���£�C�R@�R?����Q���(�A
=q                                    BxsQ2Z  T          A�@�Q�Tz��^{��\)C���@�Q�?�\)�X����ffA33                                    BxsQA   
�          A(�@�Q�@  �W���Q�C�'�@�Q�?����P�����
A��                                    BxsQO�  T          A�@�(�>B�\�N�R���?�Q�@�(�?�{�)����G�A[
=                                    BxsQ^L  �          Aff@�>����K���{@#�
@�?����"�\���Ai                                    BxsQl�  �          Aff@陚>����b�\�θR@'�@陚@���5���
A��\                                    BxsQ{�  "          AG�@�?޸R�Vff���HA[\)@�@N�R�����`(�AÙ�                                    BxsQ�>  "          A��@�33�8Q��Tz����HC�Q�@�33?�=q�;���33A<Q�                                    BxsQ��  �          A��@�����hQ���(�C���@�?�
=�N�R���
AK33                                    BxsQ��  "          A�@�
=>��
�q��֣�@(�@�
=@  �C�
��33A��                                    BxsQ�0  �          A�H@�?#�
�tz��י�@���@�@"�\�:�H��Q�A�{                                    BxsQ��  �          A�H@�p�>�33�\)��  @*�H@�p�@Q��N{��A�(�                                    BxsQ�|  T          A=q@�
=>����r�\�ָR@A�@�
=@�
�AG����A���                                    BxsQ�"  �          A\)@��>��R�l���ָR@�@��@���@  ���A�                                    BxsQ��  �          A{@�>�33�k����@0��@�@{�=p�����A�                                      BxsQ�n  �          A�@�p�>�(��p  ��=q@^{@�p�@z��>{��  A��                                    BxsR  �          A@�(�?�ff�g���z�A%��@�(�@@  ��H���RA�z�                                    BxsR�  �          A�R@���7
=�A���ffC���@�׿^�R���\��C�|)                                    BxsR+`  �          A�@�Q��%��Z�H�ď\C���@�Q��
=��Q���C�E                                    BxsR:  �          A�\@陚��\�L������C�1�@陚>#�
�j=q�ԸR?�(�                                    BxsRH�  T          Aff@�  �}p��_\)��C�!H@�  ?xQ��_\)��{@�                                    BxsRWR  �          A�H@�\)�#�
�J�H����C�b�@�\)?�G��2�\���RA6=q                                    BxsRe�  �          A�H@�33?�33�c�
�Σ�Ao\)@�33@^{��
�ip�A�=q                                    BxsRt�  "          A�
@�G�@�\�\(��Ǚ�A�{@�G�@o\)��G��F�HA�                                      BxsR�D  T          A��@�{�W
=�e��=qC�5�@�{?�Q��K���
=ALz�                                    BxsR��  �          A  @�z�>���fff���@g�@�z�@���5�����A�(�                                    BxsR��  �          A�@�>�(��h����=q@W�@�@���8������A�Q�                                    BxsR�6  �          A(�@�ff?��\)��p�@�Q�@�ff@#33�G����
A�(�                                    BxsR��  �          A(�@�G���G��mp��ׅC��3@�G�?����N�R���A`��                                    BxsR̂  �          AG�@��H��  �QG���
=C�@ @��H?Tz��S�
��@���                                    BxsR�(  �          A��@��8Q��Z=q��33C�P�@�?�{�AG����A>�H                                    BxsR��  T          AG�@�녿u�U�����C�^�@��?c�
�Vff���
@׮                                    BxsR�t  �          A��@��
���H�O\)��=qC�+�@��
?�G��A����A�                                    BxsS  T          AQ�@�=L���b�\��>���@�?�\)�@  ��\)Ab{                                    BxsS�  �          AQ�@��+��e����HC�k�@�?��\�Z=q����A�H                                    BxsS$f  �          A��@�p��5���H���C�*=@�p�?�  �w�����A<��                                    BxsS3  �          A��@�  ����s�
�ڏ\C��{@�  ?Y���{���(�@�ff                                    BxsSA�  �          Aff@�G�����S33��p�C��R@�G�>�z��k���  @(�                                    BxsSPX  �          A=q@�����Z=q��{C�4{@�>�(��mp��ҏ\@S�
                                    BxsS^�  �          A=q@�G���  �L(���\)C�t{@�G�>\)�h����(�?��                                    BxsSm�  T          A=q@�Q��(��S33���C���@�Q�>aG��mp���z�?�                                    BxsS|J  T          A�@�=q��(��dz����C�u�@�=q>\)���\��
=?�{                                    BxsS��  �          @��@ȣ��K��qG���\)C���@ȣ׿L�����
�\)C�Z�                                    BxsS��  �          @�
=@�(��~{�u��G�C��=@�(���Q����\�/�C���                                    BxsS�<  �          @�\)@�\)�|(���33� =qC�|)@�\)�����G��7Q�C�{                                    BxsS��  �          @���@�ff�g
=��
=�\)C�:�@�ff�p����
=�2�
C��                                    BxsSň  T          @�G�@�
=�X���i����C�/\@�
=��ff���
��C�                                      BxsS�.  �          @�(�@��L(����\���C���@��(����z��"�C���                                    BxsS��  �          @��H@�z��ff���
�$\)C�9�@�z�?������7�
@ʏ\                                    BxsS�z  �          @�(�@�p��=p������C��{@�p��\����$  C�#�                                    BxsT    �          @��R@����U���(�����C�Ff@��ͿB�\��Q��%��C�U�                                    BxsT�  �          A�R@�Q���������C���@�Q�ٙ������4\)C�k�                                    BxsTl  T          AG�@��\��z��r�\����C��@��\��������,\)C��
                                    BxsT,  �          A��@\��{�\���ĸRC���@\�z���\)��\C���                                    BxsT:�  T          A  @ə��z�H�c�
����C�\@ə��Ǯ������C��                                    BxsTI^  �          A�R@ʏ\�^{�s�
�޸RC��@ʏ\��ff��G���C�K�                                    BxsTX  �          A�
@�ff�`���Tz����C�,�@�ff�����z���
C�\)                                    BxsTf�  T          A�R@����p  �y����ffC��@��ÿ�  ��Q��"�C�&f                                    BxsTuP  �          AG�@��H����|(�����C�h�@��H��33�����.\)C�                                    BxsT��  �          A�R@˅�I��������ffC��3@˅�.{����z�C���                                    BxsT��  �          A33@����Q����\���HC���@��Ϳ#�
��z��$z�C��                                    BxsT�B  T          A�@�p��AG���\)�p�C��\@�p���{��z��$�C�g�                                    BxsT��  �          A(�@��R�5����H��
C��{@��R<#�
����1�
=�\)                                    BxsT��  �          A��@�z��_\)�@  ���
C��@�z��  �������C��q                                    BxsT�4  �          A  @����dz�������RC��=@��ÿh�������(�RC��{                                    BxsT��  T          A�@�{�U���
�z�C�S3@�{�����p��.33C�U�                                    BxsT�  �          A��@��\�mp���{���C��H@��\�xQ���ff�/��C�G�                                    BxsT�&  �          A��@ƸR�s33�z�H��z�C�B�@ƸR��������\C��                                    BxsU�  
�          A��@θR����   ���HC���@θR�'�����p�C��{                                    BxsUr  �          A(�@�z��{��^{����C�1�@�z��33��\)�(�C��H                                    BxsU%  T          A{@�33�c�
���
�(�C�P�@�33�c�
��=q�-�C���                                    BxsU3�  �          A�H@�{�l(��r�\��C���@�{�����(��(�C��                                    BxsUBd  �          Az�@�G�����n{��G�C���@�G���p����� �RC��                                    BxsUQ
  �          Az�@�  �hQ��\)��33C��3@�  ���������C�ٚ                                    BxsU_�  �          A�@Å��Q��}p���C�Z�@Å���R��{�#p�C�'�                                    BxsUnV  �          A  @����\)�H������C��R@���!G����\��C���                                    BxsU|�  �          A�
@�\)��\)�c�
���HC�<)@�\)��
��ff�%��C�                                      BxsU��  T          A  @����
=�W
=���RC�  @���5�����(p�C�!H                                    BxsU�H  �          A�@��R����u����C��@��R��(���=q�*��C�xR                                    BxsU��  �          A�@��
��Q��Q���=qC�%@��
�-p����H�"=qC�#�                                    BxsU��  T          A33@��������(����RC�8R@����r�\��  �=qC�+�                                    BxsU�:  �          A33@�������p��|  C��f@�������������C��q                                    BxsU��  �          A(�@�G���p����~�RC��\@�G��������=qC�)                                    BxsU�  
�          A@aG���\)?0��@��
C���@aG��Ӆ�@  ���\C��                                    BxsU�,  T          A
�\@�p����R�J=q����C��=@�p��J=q���R�ffC��\                                    BxsV �  �          A
ff@�����z��4z���z�C�*=@����_\)��  ���C���                                    BxsVx  �          A�R@�=q��Q����\��=qC�K�@�=q���H�����&{C�C�                                    BxsV  T          A�@��hQ����\���C���@���G�����&ffC�^�                                    BxsV,�  �          A��@��R�U�����C�e@��R��\���\�1�C���                                    BxsV;j  
�          A=q@����C33��\)���C��3@��׾B�\���\�0Q�C��                                    BxsVJ  T          A��@�{�G���p��#G�C�
@�{����  �C�
C���                                    BxsVX�  �          AQ�@�G��_\)��
=�/�
C��R@�G��W
=��ff�Y�C��)                                    BxsVg\  �          A=q@�p��o\)�����8��C���@�p���\)���H�h�C�+�                                    BxsVv  �          A�R@���a���ff�*�
C���@�녾�=q��ff�S��C�|)                                    BxsV��  �          A�
@��R��
=����	ffC��@��R�޸R��Q��GG�C�                                    BxsV�N  �          A
{@�Q���=q�����33C�h�@�Q�������H�==qC�<)                                    BxsV��  
�          A33@�{���
���
��33C�3@�{��\)�\�1�C�AH                                    BxsV��  
�          A
ff@�Q���{�������C�k�@�Q��
=��z��533C��H                                    BxsV�@  T          A
=@���|(���ff���C��\@�����������/��C��=                                    BxsV��  
�          A
�H@���y������C��)@�녿�ff��33�3{C��                                    BxsV܌  �          A
=q@�p��\(���z���HC���@�p��:�H��\)�&(�C�                                    BxsV�2  �          A  @�G��1���G�����C�� @�G����
��33���C��                                    BxsV��  "          A�@�(��<�����
��ffC���@�(���G���Q��\)C�R                                    BxsW~  �          A\)@�ff�n�R�������C�y�@�ff������(��'{C��R                                    BxsW$  "          A��@�\)��G��u�݅C�� @�\)�K����AffC�T{                                    BxsW%�  �          A=q@�=q��33�C�
��  C���@�=q�u���z��1Q�C�4{                                    BxsW4p  T          A ��@1����H������C���@1�������G��/�C�w
                                    BxsWC  �          AG�@E�ۅ�9������C�q@E���H���
�:�C��)                                    BxsWQ�  �          A�H@(����H�\)����C�/\@(����R����3�HC���                                    BxsW`b  
�          A=q?�Q���ff��p��d��C�c�?�Q�������
=�+��C��3                                    BxsWo  T          A=q?#�
�����
�-��C�O\?#�
��ff���  C��3                                    BxsW}�  "          A�H>�=q��\)�У��8��C��)>�=q��{��G��"{C�E                                    BxsW�T  
�          A�@<(��ڏ\�Dz�����C��@<(������  �@��C��)                                    BxsW��  "          AQ�@z����-p����
C��=@z���{��
=�9��C��                                    BxsW��  
�          A(�@xQ��ȣ��e���\)C��q@xQ��p����p��C(�C���                                    BxsW�F  
�          A�R@�����p��k���=qC�n@����hQ���ff�?  C��f                                    BxsW��  �          A(�@�G���G��l����  C��@�G��R�\��G��4\)C�q�                                    BxsWՒ  �          A	�@�Q���33�o\)��p�C�:�@�Q��*�H��\)�(ffC���                                    BxsW�8  
�          A��@�33������\��p�C�o\@�33�!G������-�RC�Z�                                    BxsW��  "          A�@�����
����ޣ�C�  @���"�\�����/G�C��                                    BxsX�  "          A
{@�33��p��e�ĸRC���@�33�%��Q���HC��                                     BxsX*  �          AQ�@�ff��p��c�
�ƣ�C�� @�ff�5�����%�\C�Ǯ                                    BxsX�  �          A�H@������y�����C��3@��"�\���\�5(�C��3                                    BxsX-v  "          A�@�����ff�1G�����C���@����y�������&
=C���                                    BxsX<  �          @�p�@��������<(����HC��H@����\����
=�%C�+�                                    BxsXJ�  �          @��@�p����R�-p�����C��@�p��`������{C�C�                                    BxsXYh  	�          @�G�@�z��Ǯ���R�0z�C��\@�z������������C�K�                                    BxsXh  "          @���@���\��  ��ffC���@�����
�q����C���                                    BxsXv�  T          @��@Fff�љ�@G�A��C��=@Fff���H��\)�(�C�0�                                    BxsX�Z  
�          @��?�G�����@���B ��C���?�G���  ?�A@��C��{                                    BxsX�   T          AG�?���Q�@���Bz�C��q?����?��A��C�7
                                    BxsX��  �          @�=q@o\)��G�?�=qAZ{C��q@o\)��(���  �2{C���                                    BxsX�L  �          @��\@S�
���?���A9��C��=@S�
�ڏ\����_�
C��                                    BxsX��  T          @�ff@H����?�z�AAG�C���@H���ᙚ��33�]��C��                                    BxsXΘ  T          A�@aG���(�@/\)A�=qC�%@aG���\�u����C�w
                                    BxsX�>  "          A�@�  ���@(�Ab=qC��\@�  ��{��33�)��C�w
                                    BxsX��  �          A��@��� z�?@  @�Q�C�l�@�����HQ���=qC�U�                                    BxsX��  "          A�?��G��%�w�C���?���Q����.33C�+�                                    BxsY	0  "          A  @'������P��C�=q@'���������C���                                    BxsY�  �          AQ�@,���
�R�0  ��
=C���@,����33��Q��-G�C��                                     BxsY&|  �          AG�@&ff�
=�S�
��z�C��\@&ff��z���z��;�RC��                                    BxsY5"  �          A�@��H�ff��
�G
=C��
@��H��ff��Q��(�C�+�                                    BxsYC�  �          A@�{� Q��33�;�
C�xR@�{��{��{�33C��                                    BxsYRn  T          A=q@�=q���B�\����C��@�=q��z����
����C�s3                                    BxsYa  T          A
=@�����ff=��
?�C���@��������5����C���                                    BxsYo�  �          A
�R@����\)?E�@��\C��@�����
��R�n�\C��=                                    BxsY~`  T          A  @�\)��=q��  �z�C���@�\)���������ffC��                                    BxsY�  �          A�
@�G���G��S�
��G�C�|)@�G����������\C��\                                    BxsY��  "          A�
@�
=��33�_\)����C���@�
=�\)�����
=C�'�                                    BxsY�R  T          A#�@ڏ\�������R���C�  @ڏ\�W�����'33C�޸                                    BxsY��  
�          A'33@�
=������Q��̣�C�f@�
=�P������"��C��R                                    BxsYǞ  	�          A)G�@�=q��=q�����
=C��@�=q�O\)���H�\)C��)                                    BxsY�D  "          A)@�p����H�z=q��{C��@�p��e��G��ffC�xR                                    BxsY��  T          A$��@�  ��ff��=q��p�C���@�  �G���33���C��3                                    BxsY�  �          A�@��H>��H���R� 
=@}p�@��H@L(�����G�A�                                      BxsZ6  �          A	�@�33>�����z��/�H@G�@�33@AG���=q���A�z�                                    BxsZ�  �          A33@���=����
���?�=q@���@%����R�	33A��                                    BxsZ�  �          A�@�=q�\(���  �#�C�)@�=q?�\�����Q�Az{                                    BxsZ.(  
�          Ap�@أ׿\)���H���C��q@أ�?�p������\Ad��                                    BxsZ<�  
�          A�H@�ff�L�����\�ffC�˅@�ff@=q��Q��
��A�                                      BxsZKt  �          A33@��?u��p��/��A�\@��@]p����H���A�                                      BxsZZ  �          @�z�@�ff?�{��33�!��A�z�@�ff@\)�k����B                                      BxsZh�  �          @�
=@�(�>�  ����@ff@�(�@�~�R��=qA��H                                    BxsZwf  T          @�Q�@�z�?�z����
���A�G�@�z�@e�Vff����B�                                    BxsZ�  �          @�(�@��R�8Q���  �C��@��R�:�H���
�%�C�0�                                    BxsZ��  T          @���@�p���z��~{�G�C�\@�p�=�G�������?��                                    BxsZ�X  �          @�  @���?�\�c33��33@�p�@���@��;�����A�=q                                    BxsZ��  �          @�33@\?���~{�  A\)@\@.�R�C�
��G�A��                                    BxsZ��  �          @���@�  ?E��vff����@�Q�@�  @(��E���\)A�Q�                                    BxsZ�J  T          @�ff@�Q�fff���\�#z�C�aH@�Q�?�����R���Ac33                                    BxsZ��  "          @��@�<#�
��  �F��>�@�@ �������*�A�\)                                    BxsZ�  T          @���@�      ���R�K{C��)@�  @
=��z��.A��                                    BxsZ�<  T          @��@�Q�J=q�����"{C��q@�Q�?�������Q�Af�H                                    Bxs[	�  "          @�@�  �Q���{��C��{@�  ?��
���\��
AZ�R                                    Bxs[�  T          @�@��׿Tz��|(���HC��@���?���x�����A2=q                                    Bxs['.  T          @�@�\)�*=q�;���C��@�\)��ff�s�
�$�HC��                                    Bxs[5�  
�          @��@@  ��=q�\)��(�C��@@  ����*�H��33C�8R                                    Bxs[Dz  
�          @��\����>�R@~{BJffCvB������ff@��A�\)C}��                                    Bxs[S   T          @�Q�?�����\?�\)A:�\C��f?�����׿���g�C��
                                    Bxs[a�  T          @�G��)��?��@��Ba\)C(L��)����=q@w�BRffCN��                                    Bxs[pl  T          @��H��녿��
@�z�B�ǮCZ�����N{@dz�B2�HCs\                                    Bxs[  �          @�z����:=q@{�BT�C�ff������@{A�ffC��=                                    Bxs[��  
�          @����u��@���Bl\)Cx
=�u����@1G�B�HC�S3                                    Bxs[�^  �          @�����
�*�H@e�BTG�C��þ��
�\)@ ��A�C��{                                    Bxs[�  
�          @���?����AG�@7
=B#�\C��H?����~�R?��HA��HC�&f                                    Bxs[��  	�          @�33����Q�@���Bt�C^� ���b�\@U�BQ�Crz�                                    Bxs[�P  	�          @��R���^{@l(�B9�C��H����  ?�\A��\C���                                    Bxs[��  
�          @�Q�@Q�����?\(�AQ�C��f@Q������R�\��C�"�                                    Bxs[�  T          @�(�@a���녽�Q�Tz�C�Ф@a���
=�����C�g�                                    Bxs[�B  
�          @�z�@�z���z��G���G�C�XR@�z��j=q��
=����C��                                    Bxs\�  
�          @ə�@����  �
=���
C�C�@����  �(Q����
C���                                    Bxs\�  �          @�Q�@�\)��(���\)�J{C���@�\)�Fff�C33��C��=                                    Bxs\ 4  T          @�(�@�=q��=q�fff�{C�K�@�=q�^�R�-p���{C�U�                                    Bxs\.�  �          @�=q@�  ��녿G���z�C�z�@�  �p���-p���=qC�E                                    Bxs\=�  T          @�
=@`  ���׿��\�33C�t{@`  ����G���C�b�                                    Bxs\L&  "          @Å@AG���
=�����!G�C�f@AG���
=�P  � Q�C�˅                                    Bxs\Z�  T          @�
=@1����׾\�`  C�c�@1���G��0���ӅC�3                                    Bxs\ir  
          @�=q@)����=q�Tz���
=C�@ @)����p��E����\C�|)                                    Bxs\x  f          @�?�����  �#�
�\C�O\?������(��ƣ�C�L�                                    Bxs\��  T          @��
@�H���
�Q���ffC��@�H�G�����:
=C��                                    Bxs\�d  �          @�\)@l(��S33�P�����C��@l(���ff��(��>=qC��H                                    Bxs\�
  �          @�(�@tz��h���#33��Q�C�1�@tz��
�H�xQ��%��C�7
                                    Bxs\��  
�          @��
@|�������\)��Q�C�@|���;��_\)�\)C���                                    Bxs\�V  
�          @�33@]p���ff�P�����C��
@]p�����Q��B��C��                                    Bxs\��  �          @ƸR@�=q��ff��z��*{C���@�=q?�R��G��2
=A
=                                    Bxs\ޢ  
�          @�{@L���{��=q�R
=C��H@L��>L�������o�R@`                                      Bxs\�H            @��
@@���,(�����J{C��@@�׾��R���H�v(�C�\                                    Bxs\��  
�          @�(�@G
=�>{��{�=��C�,�@G
=�(����Q��p  C�
=                                    Bxs]
�  
�          @�=q@��#�
����`(�C��q@��L�����\ǮC�]q                                    Bxs]:  
�          @�
=?���ff��
=ǮC�+�?�?�
=��  �B                                      Bxs]'�  �          @��\@녿�p����i��C�ff@�>�G���
=��A/�
                                    Bxs]6�  "          @�=q@u�����j=q�ffC���@u��#�
����A(�C�@                                     Bxs]E,  
(          @�
=@`���G���(����
C��@`�׿z�H�)���(�C�+�                                    Bxs]S�  �          @��
�)��?333@�(�Ba�\C%(��)������@���BY�RCK{                                    Bxs]bx  
�          @������0  @6ffB,�\Ct�쿥��l��?�33A�33Cz�                                     Bxs]q  �          @�(�����j=q?=p�@�{C]������hQ�c�
��C]xR                                    Bxs]�  T          @������>{?(�@�=qCW�
����<(��5� ��CW��                                    Bxs]�j  �          @���\(��l(�?E�A
�RCb�q�\(��j�H�^�R�p�Cb��                                    Bxs]�  T          @�{?������>.{?���C���?��������   ��=qC���                                    Bxs]��  �          @���?�
=�|(��Q��4��C�XR?�
=�N{�=q��\C�3                                    Bxs]�\  
�          @��@c33��Q�?xQ�A$��C�� @c33��G��Q��
{C��H                                    Bxs]�  �          @�G�@^{�}p���  �UG�C��)@^{�C�
�333��C�H�                                    Bxs]ר  �          @�G�@`  ��(��
=����C�P�@`  �E��p�����C�S3                                    Bxs]�N  �          @�z�@a����ÿ�
=��C�7
@a��a��qG��(�C��                                    Bxs]��  �          @�  @�33��=q=�G�?p��C���@�33��{��=q���HC��
                                    Bxs^�  �          @���@����
=���
�.{C�p�@����Q��
=��p�C�ٚ                                    Bxs^@  �          @���@�z���\)�#�
��33C��
@�z���  �Q���\)C�&f                                    Bxs^ �  �          @�p�@�Q���  ?�@�ffC���@�Q���Q��
=�j�RC�Y�                                    Bxs^/�  �          @�
=@U����R?�
=A"�\C�q@U�������4Q�C�,�                                    Bxs^>2  �          @�=q@�z����\>�(�@o\)C��{@�z���G���p��t��C�e                                    Bxs^L�  T          @θR@tz�����?.{@\C�K�@tz����R���
�[
=C���                                    Bxs^[~  T          @��
@mp����H?@  @ٙ�C�@mp���{��Q��Q��C�n                                    Bxs^j$  �          @���@y����p�>B�\?��C��@y�����\��\��
=C�                                    Bxs^x�  �          @ə�@s�
��G�����z�C��f@s�
�@  �qG��ffC�޸                                    Bxs^�p  �          @�\)@l��������
��(�C��H@l���g
=�y���Q�C��
                                    Bxs^�  T          @��@����{����C�C��3@���z�H�S�
���HC��)                                    Bxs^��  �          @��@��\��(���
=�n{C�XR@��\��G��33����C�J=                                    Bxs^�b  �          @��H@����{���R�R{C�u�@���i���R�\���\C��                                    Bxs^�  T          @���@��������!G����C��{@����xQ������(�C�3                                    Bxs^Ю  T          @�(�@�����Y����{C��3@���k��&ff���RC�O\                                    Bxs^�T  �          @�=q@���Q�O\)��=qC��@��q��%��(�C���                                    Bxs^��  
�          @��H@�����>���@9��C�  @��s33����<��C��                                    Bxs^��  �          @���@�33��\)=�?���C�*=@�33�z=q��\)�b{C�B�                                    Bxs_F  T          @�z�@�p�����<#�
=uC���@�p��r�\��Q��lQ�C��                                     Bxs_�  T          @�p�@�����H>���@!�C���@�����\��  �P  C��                                    Bxs_(�  �          @�{@���ff?�\@��
C�z�@���G���  �+�
C��                                    Bxs_78  �          @׮@�\)���    =#�
C�z�@�\)�xQ��(��l��C���                                    Bxs_E�  �          @�
=@��\��p������:{C�@��\�_\)�A���p�C�\                                    Bxs_T�  
�          @�@�\)���ÿ����%�C��@�\)�[��5����C���                                    Bxs_c*  �          @��
@�����������=qC�]q@����^{�1��ƸRC�W
                                    Bxs_q�  "          @Ӆ@�=q���\������\)C�B�@�=q�K��a���\C��R                                    Bxs_�v  �          @�33@������Ϳ޸R�v�RC���@����Tz��W����C��q                                    Bxs_�  T          @��@�(����H�\)��  C��H@�(��>{�����(�C��\                                    Bxs_��  �          @�
=@�p������
��33C�J=@�p��K��z�H�
=C�&f                                    Bxs_�h  "          @�
=@�\)��z���H����C�˅@�\)�C33�~�R�{C��H                                    Bxs_�  �          @�z�@��\��{��z��hz�C���@��\�K��L����C�S3                                    Bxs_ɴ  �          @Ӆ@�  ���׿У��d��C��@�  �P���Mp���\C�                                    Bxs_�Z  �          @�{@�  ��  ��p����C�R@�  �G
=�`�����C�j=                                    Bxs_�   T          @�@��
��=q�	����z�C�u�@��
�Fff�l���p�C�\                                    Bxs_��  �          @��
@����{�޸R�t  C�xR@���I���QG���RC�b�                                    Bxs`L  �          @��
@�
=��=q��{�`z�C�T{@�
=�Fff�E�޸RC�f                                    Bxs`�  �          @ҏ\@����33��p��,z�C�O\@���QG��0����=qC�`                                     Bxs`!�  �          @У�@����{��(��-�C��@���e�8Q���{C��                                    Bxs`0>  �          @�Q�@�
=��33�����HC�{@�
=�tz��1G���(�C��H                                    Bxs`>�  �          @���@����  ��Q��(  C��
@���j=q�7����HC��H                                    Bxs`M�  T          @�=q@�{��zΎ��<  C���@�{�o\)�C�
��Q�C��                                    Bxs`\0  �          @љ�@�Q���z�s33���C��@�Q��x���,����
=C��q                                    Bxs`j�  
�          @��H@�����ÿ�p��,��C���@���k��:=q��ffC���                                    Bxs`y|  T          @ҏ\@�������Q��n�\C���@����Z=q�S33����C���                                    Bxs`�"  �          @ҏ\@�{������4��C�N@�{�dz��:�H����C�XR                                    Bxs`��  
�          @�G�@�=q��
=�����>{C�g�@�=q�Vff�8���љ�C��{                                    Bxs`�n  
�          @Ϯ@�  �y����\)�C�C�f@�  �C33�2�\�˙�C�L�                                    Bxs`�  
�          @θR@�p��h�ÿ�  �2�\C�j=@�p��7��$z����C�}q                                    Bxs`º  �          @Ϯ@�  �^{���
�Z�\C�Ff@�  �&ff�0���ɮC��3                                    Bxs`�`  T          @�\)@���B�\��=q��
=C�Z�@����7
=��  C�s3                                    Bxs`�  
�          @�\)@�=q�Z�H����?�
C��)@�=q�(Q��$z����C���                                    Bxs`�  
�          @�G�@�
=�Mp��Ǯ�]�C��f@�
=�
=�+���{C�T{                                    Bxs`�R  "          @љ�@�p��@�׿��
�4��C�f@�p����ff��  C�                                    Bxsa�  �          @�Q�@��
�5�����dz�C��)@��
�   �$z�����C�7
                                    Bxsa�  "          @У�@�Q��H��������C��R@�Q���
�N{��p�C�T{                                    Bxsa)D  �          @�=q@���J=q�ff��(�C�xR@��� ���X�����RC���                                    Bxsa7�  �          @��H@���W���H��Q�C�H�@���(��a����C�k�                                    BxsaF�  �          @љ�@����I���33���C���@������Fff���HC��                                    BxsaU6  "          @��
@�33��\)�(Q����C���@�33�9����G���HC�!H                                    Bxsac�  T          @�(�@�{��33�*=q��(�C��)@�{�@  ���
�z�C�4{                                    Bxsar�  �          @љ�@�ff�P���n{�
=C��@�ff������{�8�HC��
                                    Bxsa�(  T          @�=q@����J�H�G���p�C�E@�����  ���H��C�                                      Bxsa��  T          @љ�@�p��Dz��XQ���\)C�\)@�p���ff�����$\)C�Ф                                    Bxsa�t  �          @�  @�  �*=q�b�\�(�C�` @�  ��\)��G��%(�C�aH                                    Bxsa�  "          @�
=@�ff��G��fff���C�5�@�ff>L���s�
���@��                                    Bxsa��  �          @˅@��ÿ��H�z=q��RC��@���>�Q����\�!��@���                                    Bxsa�f  T          @��@��������G�C���@�>�G���p��&(�@��\                                    Bxsa�  
�          @ƸR@�  ���\�X���=qC�޸@�  =����g���\?�33                                    Bxsa�  "          @�(�@�(�>���y��� �@��@�(�?����]p��(�A�(�                                    Bxsa�X  �          @�Q�@�\)�p��9����ffC�9�@�\)�u�aG��(�C���                                    Bxsb�  T          @�p�@s33�w���  �w\)C�<)@s33�AG��5���C��)                                    Bxsb�  �          @��@mp��N�R�&ff���C�s3@mp��G��g��"�C��\                                    Bxsb"J  T          @���@u��L(��  ��=qC�{@u����Q��\)C��                                     Bxsb0�  T          @��@U���=q��
=�s�C��H@U��O\)�6ff��  C��q                                    Bxsb?�  
�          @��R@Vff�r�\�G���  C��H@Vff�0���Q��Q�C�>�                                    BxsbN<  
(          @�  @��R�9����=q��G�C��)@��R� ���1G����C�<)                                    Bxsb\�  
�          @�\)@�{�G���\)��
=C�T{@�{��
=�\)���
C��=                                    Bxsbk�  �          @�
=@�Q��=q�������C��=@�Q��=q�#�
��C�Z�                                    Bxsbz.  "          @���@����
=�����C
=C�0�@����\)��G����
C�q                                    Bxsb��  �          @�33@�\)��ÿ�p��T��C��
@�\)���
��
=����C��                                    Bxsb�z  �          @�=q@R�\�*=q�>{�Q�C��\@R�\��\)�o\)�9ffC��                                    Bxsb�   �          @�(�@,(��2�\�e��*�
C���@,(����
����^ffC�N                                    Bxsb��  �          @��@Mp��O\)�7
=� p�C�g�@Mp����H�vff�6��C�P�                                    Bxsb�l  �          @�p�@��XQ��G
=��C���@�� ����(��R�
C�1�                                    Bxsb�  "          @�p�@��c�
�[��
=C�&f@���
����cC���                                    Bxsb�  T          @�ff@4z���\�~{�>(�C�k�@4z�0�������e��C�(�                                    Bxsb�^  �          @�p�@C�
�
=q�z�H�9�C�Z�@C�
�
=��{�[C���                                    Bxsb�  T          @�z�@`  �   �I����RC�=q@`  ��
=�u�8ffC��=                                    Bxsc�  T          @�z�@1��\)�|(��?(�C���@1녿(������e�C�XR                                    BxscP  �          @��@  ��G����H�m�C�\@  >�  ���\��@��                                    Bxsc)�  �          @�ff@!G���G����R�W\)C��\@!G�������t{C��                                    Bxsc8�  "          @��@AG���z������E=qC��q@AG���\)����a�\C�aH                                    BxscGB  "          @�
=@X���+��G���\C��R@X�ÿ�\)�xQ��:�
C��q                                    BxscU�  
Z          @�  @\���8���Fff�
33C�
=@\�Ϳ�=q�{��8  C��
                                    Bxscd�  �          @�\)@W
=�S�
�1G���C��@W
=�ff�q��.�RC��                                    Bxscs4  �          @��@Q��^�R�,���뙚C���@Q����qG��-Q�C���                                    Bxsc��  �          @�\)@b�\�K��,(���C��@b�\�   �i���'z�C�>�                                    Bxsc��  �          @�
=@`���A��7���{C���@`�׿�ff�p���.�\C�g�                                    Bxsc�&  �          @��R@R�\�e��H��
=C�>�@R�\�\)�b�\�"ffC�l�                                    Bxsc��  "          @�@3�
�g��7�� �C��@3�
�
=�~{�<��C���                                    Bxsc�r  �          @��@U�QG��*=q��p�C�Ǯ@U�
=�i���+{C���                                    Bxsc�  �          @�z�@w
=�8���
=�иRC���@w
=��\)�O\)��C�)                                    Bxscپ  �          @�p�@w��@���\)���C��@w����J=q�\)C�4{                                    Bxsc�d  �          @��
@����0  �   ��G�C�AH@��Ϳ�{�5� 
=C���                                    Bxsc�
  
�          @�{@���%������C���@���Ǯ�L(���
C��{                                    Bxsd�  
�          @�p�@}p��4z��
=����C�K�@}p���ff�L���33C��                                     BxsdV  
�          @�@w��?\)�ff��33C�+�@w���(��P  �33C���                                    Bxsd"�  �          @�p�@�G��C�
��\)���C�o\@�G��p��4z���C���                                    Bxsd1�  T          @��@�G��:�H�33����C��@�G��G��<���ffC��)                                    Bxsd@H  "          @�p�@R�\�;��<���G�C�,�@R�\���H�r�\�6ffC�E                                    BxsdN�  �          @��@S�
�C33�=p���C��=@S�
��=q�u��5��C���                                   Bxsd]�  �          @�p�@J=q�G
=�Fff��C���@J=q��=q�\)�>{C���                                    Bxsdl:  �          @��@\���8���,����C��@\�Ϳ�\�b�\�)�C�j=                                    Bxsdz�  �          @�33@����+����
�_
=C�Y�@�����
�Q����C�n                                    Bxsd��  �          @��@q��5��\)��{C��\@q녿�ff�S�
��\C�G�                                    Bxsd�,  �          @��
@H���4z��L(��ffC�
=@H�ÿ���~{�Bz�C��                                    Bxsd��  �          @��
@�\��R�����T�\C��
@�\�Y�����
B�C��{                                    Bxsd�x  �          @��
@Mp��[���33���C��=@Mp��Fff��G�����C��q                                    Bxsd�  �          @��@\)�}p�?�p�A��
C��
@\)����>��H@�{C�w
                                    Bxsd��  �          @���@'
=�u�?8Q�AC�*=@'
=�u�#�
����C�)                                   Bxsd�j  �          @��@aG��Fff����[
=C�K�@aG��!G������=qC�,�                                   Bxsd�  �          @�(�@)���W
=��\��C�)@)���p��C�
�!33C���                                    Bxsd��  �          @�Q�@���o\)�E��C���@��0����\)�l��C�!H                                    Bxse\  �          @��\?�녾�=q��{��C��?��?����  �z�HBG�                                    Bxse  �          @�@'��g�>���@�{C��{@'��aG��k��>ffC�Z�                                    Bxse*�  �          @�=q@ff�`  @<��BQ�C�~�@ff����?˅A���C��                                    Bxse9N  �          @��R?���n�R@P��B�
C�AH?�����?�ffA��HC�AH                                    BxseG�  �          @�ff?У��`  @^{B'�RC�u�?У���
=@�A���C�                                      BxseV�  "          @�
=@'
=�|��?�Q�A��C��
@'
=��G�>�  @:=qC���                                    Bxsee@  �          @��@H���j�H?n{A/33C�AH@H���qG���Q���C��                                     Bxses�  T          @�(�@H���p��@{A�p�C��@H������?Tz�A�C��                                    Bxse��  �          @�=q@_\)�QG�@�A�
=C�j=@_\)�r�\?c�
A��C�L�                                    Bxse�2  �          @��H@O\)���\?��A3�
C�=q@O\)��{�Ǯ���C�޸                                    Bxse��  T          @�G�@:=q��(�?�33Aw�C��R@:=q����u�333C�޸                                    Bxse�~  �          @�(�@#�
�~�R?Q�A��C�W
@#�
���ÿ\)��{C�1�                                    Bxse�$  "          @��
@(Q����׾W
=�{C��\@(Q��n{������C��
                                    Bxse��  �          @��
@
=��  ����|  C�J=@
=�U��$z�� �C��=                                    Bxse�p  �          @�ff?��R��Q��G���(�C�5�?��R�G
=�O\)�$�HC�Q�                                    Bxse�  T          @���@8Q��l(������|  C��
@8Q��B�\�{��=qC���                                    Bxse��  �          @�Q�@n{�N�R<�>��RC���@n{�C�
����C�
C�H�                                    Bxsfb  �          @�
=@s33�<(������O
=C�'�@s33�=q� ������C��{                                    Bxsf  "          @���@\)�<�Ϳc�
�"�\C�� @\)�\)������C�                                      Bxsf#�  �          @��H@.{�!�@AG�B\)C��@.{�XQ�@G�A�\)C�j=                                    Bxsf2T  �          @���@Y���<��@
=qȀ\C��@Y���`  ?��AD��C��                                    Bxsf@�  T          @�p�@vff�/\)���
�y��C�C�@vff�\)�����k�C��3                                    BxsfO�  �          @�{@�  �-p���p����C���@�  �(����R�l��C�U�                                    Bxsf^F  �          @���@�G��#33��33�}p�C��f@�G���\��z��VffC��                                    Bxsfl�  T          @���@�=q����?�(�Aa�C�U�@�=q��z�?!G�@�\)C��                                    Bxsf{�  �          @�@�33�˅@  A��C���@�33�\)?�{A��C�b�                                    Bxsf�8  �          @�z�@z=q����@Q�A�p�C���@z=q�   ?�z�A�C��)                                    Bxsf��  �          @��@R�\��@h��B033C���@R�\�8��@6ffB(�C�W
                                    Bxsf��  �          @�  @X�ÿ��@aG�B(��C�t{@X���;�@.{A�  C��)                                    Bxsf�*  �          @�ff@]p��\)@9��B�C�)@]p��S33?�
=A�=qC�1�                                    Bxsf��  �          @��
@u�n{@
=A��C�/\@u�Ǯ?�Q�A��
C��\                                    Bxsf�v  �          @��@�=q��z�?�\)APQ�C��q@�=q��Q�?�@\C��H                                    Bxsf�  T          @��@�G���Q�=��
?z�HC�l�@�G���\)�
=q�ʏ\C��{                                    Bxsf��  �          @��@������R��(���ffC�33@�����  ��ff�AC���                                    Bxsf�h  �          @�  @��H������
�p��C�%@��H�O\)�У����C���                                    Bxsg  �          @�
=@��\��33>.{?�{C��f@��\��녾W
=���C��                                    Bxsg�  �          @��R@�\)����?z�@�ffC�h�@�\)��(�=�?��C�˅                                    Bxsg+Z  �          @��@����  ?�A|��C�B�@����33?��
A5�C�\                                    Bxsg:   �          @�Q�@��׿�  ?p��A$��C�H@��׿�p�>��H@�(�C�                                    BxsgH�  T          @�
=@��׿�(�?5@���C�0�@��׿���>�\)@HQ�C�T{                                    BxsgWL  �          @�\)@��H�n{?\)@�
=C��
@��H���>u@*=qC�!H                                    Bxsge�  �          @�{@�녿�ff�8Q��(�C�&f@�녿p�׿   ����C��                                     Bxsgt�  "          @�@�zᾅ��u�'�C���@�z�.{���R�X��C��                                    Bxsg�>  "          @�p�@�=q�k���=q�A�C��R@�=q�J=q����ÅC��3                                    Bxsg��  �          @�@�녿�ff��G���C�&f@�녿u��(�����C���                                    Bxsg��  T          @�(�@��׿�G���Q�xQ�C�Q�@��׿n{������
=C��                                     Bxsg�0  �          @��@�zῷ
=�\)���C��R@�zῨ�ÿz��У�C�xR                                    Bxsg��  T          @���@��R��Q�
=��(�C�C�@��R�p�׿p���'�
C��q                                    Bxsg�|  
�          @��H@�������B�\��C��@�����z�(����  C���                                    Bxsg�"  T          @�33@��\��  �����C�^�@��\���׿(�����C�f                                    Bxsg��  �          @�33@�(�����   ��{C�Q�@�(���\)�fff�"ffC���                                    Bxsg�n  �          @�33@�=q����?��AF�\C���@�=q����?0��@��C��q                                    Bxsh  �          @��@�33�=p�?�z�A���C���@�33���?�\)AI�C�c�                                    Bxsh�  �          @��H@�p����?�
=A�ffC���@�p���G�?��\Ahz�C��                                    Bxsh$`  T          @��H@����?���AZ�RC��f@���aG�?z�HA1p�C��                                    Bxsh3  �          @�z�@�(����?�A��C���@�(���G�?�AS\)C�,�                                    BxshA�  "          @��@�  ��?\(�A�C��H@�  �O\)?&ff@�Q�C�j=                                    BxshPR  
�          @���@���aG�?���AO
=C���@���\)?�G�A8  C��q                                    Bxsh^�  �          @�\)@�G���\)?���Av{C��3@�G����?�p�AeC�/\                                    Bxshm�  "          @��R@�(�=L��?Q�A��?��@�(��W
=?L��A(�C��=                                    Bxsh|D  
�          @�(�@���>.{�B�\�33@Q�@���>aG������@.�R                                    Bxsh��  �          @��@[��@  �Y���0G�C��3@[�>��
�^{�4�@�33                                    Bxsh��  �          @�p�@dz�Q��P  �&{C���@dz�>W
=�U�,(�@S�
                                    Bxsh�6  T          @���@`�׿G��QG��)(�C��3@`��>�  �Vff�.�@���                                    Bxsh��  �          @�z�@�{�0�������{C�W
@�{=�\)�   ��ff?�G�                                    Bxshł  T          @�{@~�R�J=q�1��	�
C�Z�@~�R=�Q��8����?��R                                    Bxsh�(  T          @�z�@g��Y���J=q�!\)C�h�@g�>���P���(�@z�                                    Bxsh��  "          @�
=@��R�z�H�(����C�t{@��R�L���(Q���p�C���                                    Bxsh�t  �          @�
=@����Q������C�Y�@�����33����C���                                    Bxsi   T          @�z�@qG���=q�8�����C�@qG��#�
�E��  C�˅                                    Bxsi�  �          @�(�@y�����
�'���C��@y����G��8Q��(�C��\                                    Bxsif  �          @�33@XQ�ٙ��?\)�G�C��f@XQ�:�H�W
=�0��C��                                    Bxsi,  �          @�=q@S�
��33�E��ffC���@S�
�&ff�\(��6=qC�g�                                    Bxsi:�  �          @���@/\)��33�g��F�C�~�@/\)��  �xQ��Zz�C�aH                                    BxsiIX  �          @���@E������[��8\)C�c�@E����hQ��F�RC��                                    BxsiW�  �          @���@<(��fff�g��F�
C�~�@<(�>aG��n{�N�@���                                    Bxsif�  �          @�
=@<������4z��
=C��H@<�Ϳ�G��Vff�8��C�g�                                    BxsiuJ  T          @��R@@  ��33�C33�"C���@@  �h���^�R�@  C��f                                    Bxsi��  "          @�@5����H�^�R�B�C�l�@5����k��QC���                                    Bxsi��  �          @��@{����\)�J=qC�@{�~{��ff�~ffC��H                                    Bxsi�<  �          @��R@Dz��b�\��G��pz�C�o\@Dz��@���  �ܸRC��q                                    Bxsi��  �          @�\)@p�����Q���
=C�Ff@p�׿�=q�9���(�C�G�                                    Bxsi��  �          @�\)@�  ���H��H���C��)@�  >�  ������
@Tz�                                    Bxsi�.  T          @���@{��k��:�H��C�g�@{����
�C�
��C��3                                    Bxsi��  T          @�(�@x�ÿfff�H�����C�s3@x��=u�QG�� \)?Tz�                                    Bxsi�z  �          @�G�@<(��   �I���%�HC��)@<(���  �e�D�C��H                                    Bxsi�   �          @�Q�?�(��7
=�W��0C�C�?�(���  ����d(�C�+�                                    Bxsj�  �          @�G�@��,���`  �2  C�j=@��Ǯ��z��_z�C��                                    Bxsjl  �          @�=q@?\)� ���Mp����C��@?\)���H�s33�B�C��
                                    Bxsj%  
�          @�{@5���R�Z=q�1��C�xR@5�k��u�P��C��                                    Bxsj3�  �          @�Q�@U�O\)�c�
�7��C�8R@U>���h���=ff@�ff                                    BxsjB^  "          @�Q�@+����s�
�Q�C�>�@+��#�
�\)�`33C��
                                    BxsjQ  �          @��@L(�?���=p��"��A�Q�@L(�@�p��Q�B                                      Bxsj_�  
�          @�ff@��R��  ���(�C�` @��R>�Q��z��Ə\@��H                                    BxsjnP  �          @��R@�zᾞ�R�%���C��H@�z�>�ff�$z�����@�ff                                    Bxsj|�  "          @�z�@���p���
=��Q�C��f@���k��"�\��33C�l�                                    Bxsj��  T          @�33@W��  �!���p�C�!H@W���
=�Dz���RC�}q                                    Bxsj�B  T          @�33@p  ���R� ����
C��H@p  ����0�����C�w
                                    Bxsj��  �          @���@Tz���S33�1�HC��{@Tz�>�ff�S�
�2�\@�                                    Bxsj��  �          @�Q�?^�R?B�\��\)� B%�\?^�R?�Q��u�t33B��H                                    Bxsj�4  �          @���{?8Q���ff�C
��{?��H����t{B��                                    Bxsj��  T          @����  ?�=q����x�\C޸��  @p��g
=�Hz�B�
=                                    Bxsj�  �          @�\)���H?�  ��p���B��þ��H@,(��s�
�Yp�B�u�                                    Bxsj�&  �          @�
=�Q�?xQ����\)CB��Q�@�������n��B��                                    Bxsk �  �          @�����>�33���R¤�HC�
��?����ǮB�L�                                    Bxskr  �          @��\�Ǯ<��
��Q�ª��C1n�Ǯ?�{���z�B�\                                    Bxsk  T          @���?=p��fff���
��C���?=p�>�G���{=qA�{                                    Bxsk,�  
�          @��R?
=�E����\�RC���?
=>����
 ffB\)                                    Bxsk;d  �          @�G�?�>�(���G��fAS�?�?�ff�p  �ez�B ��                                    BxskJ
  �          @��?�G�?G������A��
?�G�?��H�z�H�f�HBQ�H                                    BxskX�  �          @���@�
?(���|(��u  A�Q�@�
?޸R�fff�T�B �                                    BxskgV  T          @�  @N�R?8Q��G
=�,��AH(�@N�R?Ǯ�1��=qA��                                    Bxsku�  �          @�@���>�G����\�?\)@�  @���?5�Y��� Q�Aff                                    Bxsk��  �          @�
=@��\�aG���ff�C\)C��
@��\=��
�����G
=?}p�                                    Bxsk�H  �          @��@��R��׿���F�\C�&f@��R�#�
�����X  C��                                    Bxsk��  �          @���@��ÿJ=q��p��pQ�C��@��þ�
=��33��p�C�Z�                                    Bxsk��  �          @�p�@^{�U��
=��
=C��@^{�E��=q�33C�+�                                    Bxsk�:  �          @���@���0  �.{��
=C�J=@����Ϳ�z���=qC��                                     Bxsk��  �          @�\)@��þ�ff�{����C���@���>�  �   ���@Tz�                                    Bxsk܆  �          @���@C33�#�
�XQ��?��C���@C33?fff�P���6�RA�33                                    Bxsk�,  "          @���@*�H�n{�j�H�Qp�C�k�@*�H=��q��Z�H@#33                                    Bxsk��  �          @�ff<��
>��H���\§
=B��q<��
?�z�����{B��q                                    Bxslx  �          @�(�?�\)��\)��=qffC�q?�\)?k���\)�A�
=                                    Bxsl  �          @��?�
=>8Q����HaH@�\)?�
=?���xQ��l��Bff                                    Bxsl%�  �          @��H@(��(���|���e�\C�u�@(�>�ff�~�R�h33A'33                                    Bxsl4j  �          @�@ff�n{�����wz�C�  @ff>�z���z�\)@��                                    BxslC  �          @���@4z�?�  �:=q�$�HA��H@4z�@�R�z�����B%\)                                    BxslQ�  �          @�@?\)?ٙ��_\)�5��A���@?\)@%�8���B#z�                                    Bxsl`\  �          @��R@k�@(Q����=qBQ�@k�@C33�����UG�B�
                                    Bxslo  T          @�ff@#�
�333�7��<�
C�W
@#�
=����<���D{@                                    Bxsl}�  �          @���?�ff�.{�r�\�E33C���?�ff�˅��(��uG�C�C�                                    Bxsl�N  �          @��@
=q�1G��`���3��C��q@
=q���H��(��`�C�˅                                    Bxsl��  
�          @���@�8���Vff�'��C�}q@�����Q��T�C��                                     Bxsl��  �          @��@>�R�����i���@{C���@>�R��{�x���QffC���                                    Bxsl�@  �          @���@fff�+��]p��-z�C��)@fff>�z��`���0@��H                                    Bxsl��  �          @�(�@_\)�J=q�j�H�6��C�� @_\)>k��p  �;��@p                                      BxslՌ  T          @��H?�{�qG�@0  B�RC�%?�{���
?У�A��
C��=                                    Bxsl�2  �          @�Q�?�z�����?�Q�A�  C�O\?�z���(�>L��@
=C��R                                    Bxsl��  T          @��R@���\)>�\)@N�RC�q�@���z�n{�-�C���                                    Bxsm~  �          @��@'
=�}p��&ff��33C��3@'
=�h�ÿ�Q���p�C��3                                    Bxsm$  �          @�z�@�z�H�����
=C�c�@�o\)�����s
=C�                                      Bxsm�  �          @��
?����33���
�{33C�^�?���w
=�����C�aH                                    Bxsm-p  "          @��?\��  ?���Aap�C���?\���ͽ#�
��\C��f                                    Bxsm<  �          @��?�{�|(�@�A�RC��?�{��p�?�p�Aj�RC���                                    BxsmJ�  �          @�p�@Vff=�Q��Z�H�6=q?�=q@Vff?�  �QG��,\)A��R                                    BxsmYb  T          @�\)@Vff?�Q��^�R�1p�A��\@Vff@z��B�\��A��
                                    Bxsmh  T          @�p�@hQ�?B�\�L(��"�\A=�@hQ�?�=q�7���A�(�                                    Bxsmv�  T          @�{@u?�(��5��z�A���@u?�
=�=q��Q�A�33                                    Bxsm�T  �          @�@J�H?=p��i���A�AQp�@J�H?��Tz��+\)A��H                                    Bxsm��  �          @�Q�@{��s33�%�ffC�,�@{��u�0  ���C�E                                    Bxsm��  �          @�{@i���
=q�{��  C���@i����
=�=p��
=C�K�                                    Bxsm�F  �          @���@e������:=q��RC�#�@e��@  �N{�%�\C�3                                    Bxsm��  �          @�ff@|�Ϳ�
=�(����C��
@|�Ϳ���'��
=C��R                                    BxsmΒ  "          @�{@~�R��ff��\�ݮC���@~�R�����+���C��                                    Bxsm�8  �          @�  @�33��33�Q���{C���@�33���
�#33��=qC�Y�                                    Bxsm��  �          @�Q�@�p��33��\)���C��\@�p���p������C�33                                    Bxsm��  "          @�
=@��
��z��p���(�C���@��
��=q�=q���C�{                                    Bxsn	*  "          @��@a��33��R���
C�|)@a녿����?\)�
=C�f                                    Bxsn�  T          @�(�@_\)���!���(�C�o\@_\)����B�\��C��                                    Bxsn&v  T          @���@K��"�\�+��\)C��=@K���G��P  �'33C��f                                    Bxsn5  "          @�G�@E�!��&ff�Q�C�T{@E��\�J�H�&C�)                                    BxsnC�  �          @�@r�\?u�!G��33Ab�\@r�\?�=q����{A���                                    BxsnRh  T          @�=q@h�ÿ�Q��-p��G�C��@h�þ�ff�;��z�C��H                                    Bxsna  
�          @��
@#�
�L���(�����C�U�@#�
����L(��'�HC�R                                    Bxsno�  �          @�=q@Y�������z�C��q@Y���У��=p���C�33                                    Bxsn~Z  T          @���?�ff���׿z�H�8��C�p�?�ff����
=��\)C�K�                                    Bxsn�   "          @�(�?�33��(����R�mp�C�P�?�33�z=q�����C�l�                                    Bxsn��  T          @�?�ff��Q쿽p���ffC�f?�ff�~{�%���RC��                                    Bxsn�L  �          @�z�?�Q���녿޸R��C�=q?�Q��n{�333�
=C��R                                    Bxsn��  T          @�z�?��R��33���
���HC�p�?��R�s�
�%�z�C���                                    Bxsnǘ  T          @�(�?��R��(���
=����C��?��R�s33�/\)�	C��                                    Bxsn�>  
�          @�z�@ �����
�G��  C�^�@ �����׿����G�C�K�                                    Bxsn��  �          @��?�\)���׿!G���RC��R?�\)���R��\���C���                                    Bxsn�  �          @�\)@5��u��33����C�4{@5��Vff����(�C��                                    Bxso0  
�          @���@*=q��Q쿀  �<  C�˅@*=q�g����R��G�C�*=                                    Bxso�  �          @�\)@���  ��ff�B�HC�b�@��u�ff��{C���                                    Bxso|  T          @�Q�@����zῷ
=��p�C�3@���hQ��(���  C��)                                    Bxso."  	�          @��@��ff��G��k�C��=@�o\)��\�ۮC���                                    Bxso<�  
�          @��@	����녾����z�C���@	��������ff��G�C�@                                     BxsoKn  �          @�G�@   ���
<#�
>#�
C��f@   ��\)�����IG�C�N                                    BxsoZ  T          @�
=@%����=��
?aG�C��f@%����
��  �:ffC��                                    Bxsoh�  �          @�z�@{����(���33C�C�@{�|(���p���G�C��                                    Bxsow`  
�          @�=q@>�R�mp��@  �  C�k�@>�R�X�ÿ���  C���                                    Bxso�  �          @�=q@HQ��fff�5��HC�}q@HQ��S33�˅��z�C���                                    Bxso��  
�          @���@B�\�n{�h���+�
C��q@B�\�W���������C�                                    Bxso�R  �          @��@,���p  �����ip�C�޸@,���U��
=���
C���                                    Bxso��  �          @��H@Vff�\�ͽ#�
���HC��@Vff�U��h���.ffC���                                    Bxso��  "          @�z�@Dz��p  �\)��z�C���@Dz��_\)���R��=qC��                                    Bxso�D  �          @�z�@G��l�Ϳ��ƸRC�{@G��\�Ϳ�Q����C�3                                    Bxso��  �          @��H@333�h�ÿ�������C��@333�L(��{����C��                                    Bxso�  �          @��@S33�c33���
�z�HC�z�@S33�Z=q�xQ��8��C��                                    Bxso�6  �          @�33@I���j�H>��?���C�U�@I���e�E���RC��                                    Bxsp	�  "          @�\)@S33�Q녿�R��(�C���@S33�AG�����p�C��                                     Bxsp�  �          @�p�@Fff�P�׿aG��1p�C��=@Fff�;�������C�J=                                    Bxsp'(  �          @�
=@AG��c�
�W
=�&ffC�"�@AG��X�ÿ���X��C�Ф                                    Bxsp5�  �          @�@<���e=�?�  C��{@<���`�׿G��=qC��                                    BxspDt  
Z          @��@3�
�h��>�=q@S�
C��@3�
�e�&ff�{C���                                    BxspS  T          @�{@HQ��]p����R�s33C�
@HQ��QG���z��f{C��                                    Bxspa�  T          @��R@I���S�
��  �EC���@I���=p���ff��ffC�g�                                    Bxsppf  �          @��R@R�\�Mp��n{�7\)C��)@R�\�8Q�ٙ���{C�k�                                    Bxsp  �          @���@N�R�Z=q�Tz�� ��C���@N�R�Fff��z���ffC�R                                    Bxsp��  T          @�ff@7
=�g��0�����C�+�@7
=�U�������p�C�S3                                    Bxsp�X  �          @��H@%��A�����Q�C�0�@%���H�0����C�c�                                    Bxsp��  �          @��@%��-p��Fff���C��@%���{�k��D�\C�"�                                    Bxsp��  T          @�p�@0  �7��7��Q�C���@0  �z��_\)�5��C���                                    Bxsp�J  �          @��@(���8Q�����C�B�@(����R�:�H� �C��                                    Bxsp��  �          @��R@A��K�=�G�?�z�C��@A��G
=�.{��RC�!H                                    Bxsp�  �          @�@0���l(��:�H��C�c�@0���X�ÿ�\)���HC���                                    Bxsp�<  �          @�{@A��S�
��(��t  C�<)@A��:=q�   ��  C��                                    Bxsq�  T          @���@*=q�(Q�?�A�33C��=@*=q�AG�?��HA��C��                                    Bxsq�  T          @�Q�@P  �E��aG��1�C�J=@P  �;��xQ��G�C���                                    Bxsq .  T          @�z�@^�R�0  ��Q��p��C��
@^�R����=q��(�C��)                                    Bxsq.�  �          @�p�@q��
=�������C���@q녿��H������
C�K�                                    Bxsq=z  �          @�{@xQ��   ���H��33C�aH@xQ�\�	����=qC�S3                                    BxsqL   �          @��
@AG��.�R�
=����C��=@AG��Q��-p���C�`                                     BxsqZ�  �          @��R@R�\�G��*=q�	��C�=q@R�\����Dz��#�\C�,�                                    Bxsqil  �          @��@U����R�(�C���@U��G��3�
�\)C��\                                    Bxsqx  T          @��R?���z�H�:�H�  C��=?���hQ������C���                                    Bxsq��  �          @QG��n{�5�?��A��C{�R�n{�AG�>�
=@�  C|��                                    Bxsq�^  �          @Vff���H�*�H?�Q�A���Cu�׿��H�<(�?=p�AQ��Cw��                                    Bxsq�  �          @����%��ff@g
=BH�CR�q�%���@G�B&Q�C^�R                                    Bxsq��  �          @�Q������G�@J=qB;(�CXB������R@(��BCa�                                    Bxsq�P  �          @�G�����33@>{B=��Cd� ����-p�@Q�B�ClxR                                    Bxsq��  �          @9���fff����@�\B3
=CtͿfff��\?��
A���Cx��                                    Bxsqޜ  S          @L(����@�B$�Cp�῕�!�?�G�A㙚Cu33                                    Bxsq�B  T          @QG�?^�R�0  ?�
=A�(�C�Ф?^�R�AG�?5AL��C��                                    Bxsq��  
�          @n{?�G��:=q�}p���ffC���?�G��%���z���RC��                                    Bxsr
�  �          @Vff>���������C���>���\)�'��V�C�<)                                    Bxsr4  �          @��?&ff����?8Q�A��C��R?&ff��p�>�  @��C�+�                                    Bxsr'�  T          @���@���/\)��(����RC���@���
=��������C�3                                    Bxsr6�  �          @�=q@
=�QG���G����RC��@
=�333�����p�C��                                    BxsrE&  �          @���@��W
=��
=����C�ٚ@��J=q��(����C���                                    BxsrS�  �          @Fff?����%�?z�HA�33C��q?����/\)>��
@�  C�>�                                    Bxsrbr  T          @`  ?˅�ff@��BffC���?˅�$z�?�\)A�=qC��                                    Bxsrq  �          @�  @\)�G�@
=B (�C��\@\)�.{?�  A���C�C�                                    Bxsr�  �          @��
@=p���@#�
B�HC��@=p���R@G�A��HC��                                    Bxsr�d  T          @�G�@\(����@{A�ffC���@\(���?ٙ�A��C��)                                    Bxsr�
  "          @��@XQ��Q�@��AᙚC��@XQ��%?���A���C�O\                                    Bxsr��  T          @��?���Vff?�\)A���C���?���h��?B�\A(��C���                                    Bxsr�V  "          @��@?\)�AG�?(�A33C�` @?\)�E������C��                                    Bxsr��  �          @��H@Vff�*=q=�Q�?�(�C�˅@Vff�&ff������C�R                                    Bxsrע  �          @�{@\(��.{�����(�C��
@\(��"�\����]�C���                                    Bxsr�H  �          @��@w��녿z�H�K33C�*=@w���p���Q���z�C���                                    Bxsr��  �          @��
@z=q��33�������C�%@z=q��ff��{����C��                                    Bxss�  �          @���@s�
��녿��R���\C���@s�
�@  ��(���p�C�o\                                    Bxss:  T          @��\@z�H��\)��z���z�C�@z�H�@  �����Q�C���                                    Bxss �  �          @��@�녿z῕�}�C���@�녾�\)���\���RC�                                    Bxss/�  
�          @�Q�@\)?B�\��\)�v{A,(�@\)?�  �fff�E�Ab{                                    Bxss>,  �          @�(�@x��?�G�������{A�@x��?�  �L���)A�{                                    BxssL�  �          @�
=@k�?\(��������ARff@k�?�G����
��  A�                                    Bxss[x  �          @��\?Ǯ��p��z��Q�C��?Ǯ���
�����(�C��)                                    Bxssj  �          @�\)�#�
��ff?   @�p�C����#�
��{�z���RC���                                    Bxssx�  T          @��\�����?�z�A`  C^�����z�=L��?\)C��                                    Bxss�j  �          @�=q��33��z�?��
A���Cw�R��33��?�R@�Cx�                                    Bxss�  �          @��Ϳ�Q���(�@�\A�
=Ctٚ��Q����?�p�A_\)Cv��                                    Bxss��  �          @��
��\��=q?���A�\)Ct���\��(�?=p�A��CvE                                    Bxss�\  �          @�ff��{���\?���A�33Cw�H��{���\>�ff@�Q�Cx�                                    Bxss�  �          @�ff����G�?��A�
=Cw\������?   @��Cx0�                                    BxssШ  �          @�  � ����33?��A�Q�Cv^�� �����\>Ǯ@�G�Cwff                                    Bxss�N  
�          @�G����H����?�(�A��Cu�R���H��(�>�p�@�  Cw�                                    Bxss��  T          @�33��G����?L��A z�Cw#׿�G���\)�aG��/\)Cwz�                                    Bxss��  T          @��\�У����>�z�@fffCx�У����=p����Cx��                                    Bxst@  �          @�(������=q=u?G�Cy:����~{�k��B=qCx�q                                    Bxst�  �          @�  �G���p�?�\)Ag\)Cw���G���33>#�
?У�Cx^�                                    Bxst(�  
�          @�\)�   ���?�z�A�\)Cv� �   ��\)?=p�@���Cx)                                    Bxst72  �          @�=q�����@ ��A�  C{Ǯ�������?W
=A�C|��                                    BxstE�  �          @�녿�
=��  @p�A��Cy�\��
=����?���A;�C{�                                    BxstT~  
          @����
�H���
@��A�(�Cr5��
�H���\?���AnffCt�)                                    Bxstc$  	�          @��\������@%A��Cp^�������?�ffA�ffCs.                                    Bxstq�  	�          @��H�ٙ�����?��HAw\)Cw.�ٙ���ff>W
=@+�Cx                                    Bxst�p  �          @��
���x��?@  A�Cv� ���}p��W
=�/\)Cw
                                    Bxst�  �          @�G��������\?�\)Ae��Cx�Ϳ�����\)=�?��HCy=q                                    Bxst��  T          @�zΰ�
���
?8Q�A�RC|���
��p���z��qG�C|��                                    Bxst�b  �          @�{����
=?�\@ӅC~������
=��\��ffC~��                                    Bxst�  "          @���p������>���@���C��׿p������!G��33C���                                    Bxstɮ  �          @�{��G��j�H?�  A��Cth���G��\)?Q�A*=qCv+�                                    Bxst�T  T          @��\��z����?��RA}�C{쿴z���G�>aG�@1�C{�=                                    Bxst��  �          @�\)=��
���R�   �ڏ\C��\=��
�}p����R��
=C��
                                    Bxst��  "          @�(�?c�
���ÿ�(���=qC��?c�
�l(��<(����C��\                                    BxsuF  "          @�{?����A��G��Q�C�?�������;��9  C���                                    Bxsu�  "          @��?ٙ��2�\����=qC��\?ٙ��Q��B�\�@�\C�T{                                    Bxsu!�  �          @��?�ff�p  ��G�����C�Ф?�ff�Vff�����C�%                                    Bxsu08  T          @�G�?�z��z�@K�B@Q�C�Ǯ?�z��AG�@"�\B�C�l�                                    Bxsu>�  "          @�z�?��
�Vff@
�HA���C���?��
�qG�?��A�{C���                                    BxsuM�  �          @���?�(��^�R?���A���C��f?�(��l��>��H@�33C�f                                    Bxsu\*  
�          @��@=p��8Q�?=p�A!�C��@=p��>{<�>��C�s3                                    Bxsuj�  
�          @���@(��b�\�L���-�C���@(��P  ��\)��p�C���                                    Bxsuyv  T          @���?�
=����p���@��C���?�
=�tz��z���ffC�K�                                    Bxsu�  "          @��\?��
�~�R��G����\C�G�?��
�c�
����
=C��                                    Bxsu��  �          @�  >���_\)?.{A/�
C�Ф>���c33�.{�,(�C��                                     Bxsu�h  �          @��H?��
����?��A��\C���?��
��ff>k�@8��C�~�                                    Bxsu�  �          @��R?�  ��Q�?�Q�A�ffC��q?�  ��G�?�R@��HC�9�                                    Bxsu´  �          @��?J=q��p�@
�HA�  C�
=?J=q���?�ffAB=qC���                                    Bxsu�Z  �          @�z�?\(���33@%A�{C���?\(����H?�(�A��C��                                    Bxsu�   "          @��?&ff���@#33A��
C�1�?&ff��z�?�A�z�C���                                    Bxsu�  
�          @���?(�����
@ ��A�{C�\?(�����H?��AiG�C���                                    Bxsu�L  
�          @��\?����8Q�@l��BI  C�0�?����k�@9��B�C�)                                    Bxsv�  
�          @��H@ ���'
=@*=qB�HC��\@ ���J�H?��HA�
=C�1�                                    Bxsv�  �          @�p�@"�\�]p�@��A�RC�,�@"�\�{�?��RA���C�xR                                    Bxsv)>  �          @���?��R�Z=q@%�B�RC�  ?��R�{�?�Q�A�=qC�n                                    Bxsv7�  "          @���@   �c33@@  Bp�C���@   ���@�
A���C��\                                    BxsvF�  �          @�\)?�Q��W�@L(�B!
=C�Q�?�Q���G�@33A�p�C�]q                                    BxsvU0  �          @���@AG��HQ�?��HA�=qC��)@AG��`��?�z�A`��C�\)                                    Bxsvc�  �          @�(�@K��$z�>�@�\)C���@K��&ff�#�
���C�]q                                    Bxsvr|  "          @hQ�@!G����?�ffA�Q�C�4{@!G��%�>�G�@ᙚC�(�                                    Bxsv�"  �          @x��@C33���?��HA��
C�&f@C33��?xQ�Ah��C�9�                                    Bxsv��  �          @w�@1녿�?�A�C���@1��33?�ffA��\C�33                                    Bxsv�n  
�          @|��@8�ÿ�ff@ ��A�z�C��@8����R?\A���C�/\                                    Bxsv�  "          @{�@:=q��
?��HA���C�O\@:=q�=q?�A�33C�&f                                    Bxsv��  
�          @��@Dz��@��A��C��q@Dz��=q?�p�A���C��
                                    Bxsv�`  T          @�=q@0���-p�@QG�B �C���@0���Z=q@!�A�C��H                                    Bxsv�            @��@u��?xQ�A@(�C��q@u�%>�Q�@��C�f                                    Bxsv�  �          @�
=@��*=q�#�
��ffC��f@��$z�.{���\C�4{                                    Bxsv�R  �          @�Q�@����&ff�k��'
=C�Q�@����{�Y����C��
                                    Bxsw�  �          @��R@�{�(�>��?�G�C��q@�{�
=q�\���
C��                                    Bxsw�            @�33@����p�>L��@�C�U�@����(���{����C�n                                    Bxsw"D  �          @���@�z���>\@�33C�XR@�z��ff�B�\�  C�8R                                    Bxsw0�  �          @��H@s33��R?�(�AxQ�C��=@s33�p�?&ffA33C���                                    Bxsw?�  �          @��@vff��R?s33AA�C���@vff���>Ǯ@�z�C�                                    BxswN6  �          @�G�@w
=�p�?(��A
=C�!H@w
=�33=�G�?�z�C��)                                    Bxsw\�  �          @���@�{�{��=q�N{C��@�{��J=q�C���                                    Bxswk�  �          @��R@���z�=�?�G�C���@����\�\��p�C���                                    Bxswz(  �          @��@q녿��?�Q�A��HC�5�@q녿�\?ǮA�Q�C�o\                                    Bxsw��  �          @��@mp����@=qA�33C�3@mp����@G�AθRC���                                    Bxsw�t  �          @�ff@s33��z�@��A��
C��H@s33��(�@ffA�=qC��{                                    Bxsw�  �          @��@p���Q�?�\)A��\C�:�@p���p�?��ATz�C�h�                                    Bxsw��  �          @��
@a����?�Q�A\)C�\@a���H?�RA��C�˅                                    Bxsw�f  �          @�{@\)�G�?��A��RC��3@\)��?J=qA33C�%                                    Bxsw�  �          @��R@�(��{?�Q�A^ffC���@�(��,(�?\)@�
=C�w
                                    Bxsw�  �          @��@����
=������C�  @��׿�Q�u�A�C�q                                    Bxsw�X  �          @��@���p��n{�4(�C��@������
=��ffC���                                    Bxsw��  �          @��@�G����?fffA.=qC��
@�G��"�\>���@eC��3                                    Bxsx�  �          @�ff@�������8Q���
C�g�@�����\�@  ���C���                                    BxsxJ  T          @�
=@��'
=�����C��@������33�V{C��                                    Bxsx)�  �          @���@����.�R����n{C��{@����������C��
                                    Bxsx8�  �          @���@q���R������G�C�Y�@q����Q�����C���                                    BxsxG<  �          @��H@I���>{��(���=qC�S3@I���!��	����=qC��H                                    BxsxU�  �          @�(�@HQ��>{����\)C�C�@HQ��{���33C��R                                    Bxsxd�  �          @��\@7��N�R��(����RC��R@7��1G��p���{C�                                    Bxsxs.  �          @�33@0���Q녿�=q���C��@0���333�����C�O\                                    Bxsx��  �          @�z�@w���=q�����C�Z�@w��������z�C�w
                                    Bxsx�z  �          @��\@i���
�H����\)C��@i���ٙ������=qC���                                    Bxsx�   �          @�G�@Z=q�p���(����C��@Z=q��(��p���{C���                                    Bxsx��  �          @�Q�@c�
��\)��\��G�C��@c�
�����(���
C��\                                    Bxsx�l  �          @��@W�����
���C��@W���Q��+��z�C�N                                    Bxsx�  �          @�\)@5��녿�Q����C�'�@5���
=��33��G�C��f                                    Bxsxٸ  �          @|(�@'
=�7
=?B�\A1p�C�9�@'
=�=p�=#�
?\)C��q                                    Bxsx�^  �          @���@'
=�7
=?��A���C�5�@'
=�C33>���@��C�J=                                    Bxsx�  
�          @���@7��e?��HAk�C�O\@7��q�>�z�@aG�C���                                    Bxsy�  �          @�G�@H���_\)?h��A/�C�  @H���g
=<�>�{C���                                    BxsyP  �          @���@L(��I��?
=q@ۅC��\@L(��K���  �J=qC��                                    Bxsy"�  y          @�(�@i����ff��H��\)C�+�@i���+��*=q��\C�Ǯ                                    Bxsy1�  �          @�(�@mp����z���33C��R@mp���  ���� \)C���                                    Bxsy@B  �          @�p�@k������
=��=qC�E@k��Q��(����C���                                    BxsyN�  �          @�=q@q녿G��(���C�33@q녾���#33�{C��R                                    Bxsy]�  �          @�G�@n�R�aG���H� �HC�U�@n�R����#�
�	�C��q                                    Bxsyl4  �          @��@Y������,(����C��q@Y����G��8��� �C�N                                    Bxsyz�  T          @�
=@`�׿��Q�� ffC��@`�׿J=q�)����C��                                    Bxsy��  �          @��@?\)�#�
�A��2�
C��{@?\)>��E�7�R@=q                                    Bxsy�&  �          @�33@��>�{�e�_��A ��@��?��H�X���Nz�A���                                    Bxsy��  �          @��\@ �׾�
=�^�R�W�C�B�@ ��>�ff�^�R�VA#�                                    Bxsy�r  �          @�G�@ �׿�G��?\)�:�HC��@ �׾��H�Mp��M��C��f                                    Bxsy�  �          @�=q?��H�^{?�p�A���C�!H?��H�j=q>���@�C���                                    BxsyҾ  �          @��H@G��c�
?s33AJ=qC�B�@G��l(�=#�
>��C���                                    Bxsy�d  �          @�(�?�G��Z�H��
=��
=C��
?�G��=p���R��HC�XR                                    Bxsy�
  �          @z�H?�(��_\)��\)����C�!H?�(��S�
������C��q                                    Bxsy��  �          @�Q�?�ff�p  ��G����C��{?�ff�fff���
�q�C��                                    BxszV  �          @���?�p��^{�\��ffC��
?�p��QG���p����C���                                    Bxsz�  �          @���@��fff��z��\)C��)@��Z=q��
=���
C�c�                                    Bxsz*�  �          @���?�Q��g
=>��@33C�"�?�Q��a녿B�\�*=qC�e                                    Bxsz9H  T          @�
=?���3�
�:=q�"�\C��?�녿��R�`���O�
C�Ǯ                                    BxszG�  �          @�p�?@  �8Q��8���0�\C�O\?@  �z��aG��g�HC��)                                    BxszV�  �          @y��>�=q�/\)�,���1�RC��\>�=q��p��S33�k=qC��H                                    Bxsze:  T          @u?�
=�L(���\)��G�C�{?�
=�0  �Q��	�HC�                                    Bxszs�  �          @|��?޸R�G��Ǯ��
=C���?޸R�(Q���\�33C���                                    Bxsz��  �          @y��@��� �׿�����\)C�  @����\�	���G�C�)                                    Bxsz�,  �          @u�@\)�
=��z����C�E@\)��\)�����C��\                                    Bxsz��  �          @vff@
=q�&ff��
=��
=C��
@
=q�ff���p�C��=                                    Bxsz�x  �          @s�
@Z�H�B�\������C��R@Z�H�Ǯ�����z�C��                                    Bxsz�  �          @y��@N{���\��33��G�C�4{@N{��\�ff�(�C���                                    Bxsz��  �          @z�H@Fff�(����\)C�t{@Fff�#�
�
=�33C��H                                    Bxsz�j  �          @{�@?\)����R��
C��@?\)=��!�� �R@                                    Bxsz�  �          @x��@G
=�����\�C�=q@G
=>8Q��z���H@W
=                                    Bxsz��  �          @u@N{���\��G���  C�9�@N{�
=q���H���C�=q                                    Bxs{\  �          @w�@C�
����(���\)C���@C�
���׿����C��q                                    Bxs{  �          @z�H@*�H�,�Ϳk��Z=qC�T{@*�H�Q��=q��Q�C�%                                    Bxs{#�  �          @~{@G
=��������C�0�@G
==u�z����?�Q�                                    Bxs{2N  �          @|��@J�H����\)��C��)@J�H�#�
�z���
C��                                     Bxs{@�  �          @g�@���L��� ���2��C��f@���\)�(Q��>
=C�]q                                    Bxs{O�  �          @z�H@<�;�ff���C��@<��>.{�=q���@N�R                                    Bxs{^@  T          @�ff@hQ�>W
=�
=����@S�
@hQ�?=p���p���A8��                                    Bxs{l�  �          @�z�@fff>��R��p����@�ff@fff?O\)��=q�ӮAJ=q                                    Bxs{{�  �          @fff@;�=u���R��\?�{@;�?\)��z��Q�A.{                                    Bxs{�2  �          @l(�@>�R<#�
��
�
�>u@>�R?
=q���R�
=A$                                      Bxs{��  �          @j�H@,�;aG����'z�C���@,��>������%z�A	�                                    Bxs{�~  �          @u@'
=����333�;��C�XR@'
=?��/\)�7(�AF=q                                    Bxs{�$  �          @q�@�ͽ����7
=�EQ�C��H@��?#�
�2�\�>Ajff                                    Bxs{��  �          @tz�@1G����
�$z��*�
C��R@1G�>�33�$z��*z�@�G�                                    Bxs{�p  �          @u�@+����ÿ�������C���@+���p����
��ffC��\                                    Bxs{�  �          @�z�@4z��8Q�#�
��C�>�@4z��1녿@  �+�C���                                    Bxs{�  �          @�z�@O\)�(��G��.�HC���@O\)�	���������C�0�                                    Bxs{�b  �          @�@W
=�
=�5�{C�w
@W
=���ff���C��                                    Bxs|  �          @�ff@S�
�{�8Q��=qC���@S�
��Ϳ������C�*=                                    Bxs|�  �          @�  @^{�Q�z����HC���@^{�	����
=���RC��                                    Bxs|+T  �          @�Q�@c�
��\�Ǯ���C���@c�
�
=�z�H�T(�C���                                    Bxs|9�  �          @�
=@`���  ��Q���{C��=@`�����n{�N{C���                                    Bxs|H�  �          @�p�@Tz��ff�333�z�C�P�@Tz��������C��                                    Bxs|WF  �          @�p�@fff��(��B�\�(Q�C��@fff���H��  ��\)C�T{                                    Bxs|e�  �          @�@e��   �5�C�e@e���  ���H��G�C�                                      Bxs|t�  �          @�Q�@W�����33��C���@W���(��˅��G�C�:�                                    Bxs|�8  �          @~�R@Q녿�zῑ���=qC�ٚ@Q녿�ff�������C�Y�                                    Bxs|��  �          @z=q@K���(���������C�%@K���{��=q����C���                                    Bxs|��  �          @x��@G�����=q����C��R@G���
=��\�ٮC���                                    Bxs|�*  �          @�Q�@\)�=q���R��z�C���@\)��ff�#33���C��                                    Bxs|��  �          @��\@��Q��z���\C�33@���
=�7��3��C��                                    Bxs|�v  �          @�\)@<(��!녿�\)���C���@<(�� ����R���C��3                                    Bxs|�  �          @�
=@HQ��%��ff�g
=C�8R@HQ��{��Q���=qC�U�                                    Bxs|��  �          @�\)@<���<�Ϳ\)���
C�y�@<���,�Ϳ�����{C��H                                    Bxs|�h  �          @�
=@6ff�B�\����=qC���@6ff�1녿�{����C��{                                    Bxs}  �          @��@:=q�<�;�z�����C�N@:=q�1G�����mp�C�8R                                    Bxs}�  �          @�{@7��A녾W
=�:�HC��
@7��7���G��_�C���                                    Bxs}$Z  T          @�p�@:�H�333>��
@�G�C��@:�H�2�\��G����C�%                                    Bxs}3   T          @���@(Q��J�H>�G�@��HC��q@(Q��J�H��
=���C�ٚ                                    Bxs}A�  �          @�
=@\)�Y��>��@`  C��@\)�U�(����C�S3                                    Bxs}PL  �          @�ff@/\)�K�=���?��C�aH@/\)�E�@  �$��C�Ǯ                                    Bxs}^�  �          @�Q�@*�H�R�\����{C���@*�H�H�ÿ}p��Y�C�7
                                    Bxs}m�  �          @�33@=p��2�\>�p�@���C�\)@=p��2�\�Ǯ��G�C�aH                                    Bxs}|>  �          @��@C�
�.�R>�\)@w�C�
@C�
�-p������Q�C�9�                                    Bxs}��  �          @���@5��L(�=L��?#�
C��@5��E��O\)�/\)C�G�                                    Bxs}��  �          @�@;��=p�>�?�G�C�Y�@;��8�ÿ+����C��
                                    Bxs}�0  �          @�\)@<(��B�\    �#�
C��@<(��;��Q��3\)C��\                                    Bxs}��  �          @�
=@+��O\)>��?��RC��
@+��J=q�:�H� Q�C�1�                                    Bxs}�|  �          @�p�@.�R�HQ�>#�
@	��C���@.�R�C33�333�{C��=                                    Bxs}�"  �          @��\@$z��G�=�\)?�G�C�@$z��AG��G��0��C�4{                                    Bxs}��  �          @�  @ ���Dz���ǮC���@ ���<�Ϳ\(��H(�C�1�                                    Bxs}�n  �          @}p�@%�>�R�����C�y�@%�5��u�_�C�9�                                    Bxs~   �          @xQ�@)���333�B�\�0��C��H@)���(�ÿp���aG�C��3                                    Bxs~�  �          @~{@C�
�33���
�pz�C��@C�
��Q��\)��z�C��R                                    Bxs~`  T          @|(�@9���(����
�r�RC���@9���z��z���\)C�>�                                    Bxs~,  �          @���@B�\�������(�C��@B�\��(���  ��(�C��f                                    Bxs~:�  �          @�Q�@A����aG��L��C��@A����G���z�C���                                    Bxs~IR  �          @���@���HQ�@  �+\)C��=@���333��������C�Q�                                    Bxs~W�  �          @��@���Vff�(��
�\C���@���C33��G����\C���                                    Bxs~f�  �          @���@��N{�J=q�4��C��)@��8Q������HC�%                                    Bxs~uD  �          @{�@��=p���G��o33C�W
@��$z������HC�\)                                    Bxs~��  �          @~{?�\)�XQ�&ff�"ffC��?�\)�C�
�Ǯ��33C�H                                    Bxs~��  �          @�(�����=q>�ff@�=qC�������ÿ0���\)C�
                                    Bxs~�6  �          @�(�    ���
�L���1�C�    �y����=q����C�f                                    Bxs~��  �          @��H>��������z����HC��3>���s�
��33���C��\                                    Bxs~��  �          @��
��p���녽u�\(�C�l;�p��x�ÿ�Q���\)C�O\                                    Bxs~�(  �          @��
?O\)�}p���
=���
C��\?O\)�k���  ��\)C�9�                                    Bxs~��  �          @�z�?�  �j=q�O\)�5��C�˅?�  �R�\��ff��z�C��                                    Bxs~�t  �          @�p�?����g
=�xQ��W�C��3?����L(���Q����
C�
=                                    Bxs~�  �          @�?���o\)����g�C�33?���S33�33����C��                                    Bxs�  �          @�p�?�G��dzῙ����(�C�{?�G��Fff�	����\)C�Ǯ                                    Bxsf  �          @��
@z��U���G���{C���@z��5�����ffC�
=                                    Bxs%  
�          @��?��Z�H�������
C���?��9�������RC��H                                    Bxs3�  �          @�ff?����g
=����ffC��?����Dz��Q��
33C��                                     BxsBX  �          @��
?�(��`  ���
���C�Y�?�(��;��p���\C�J=                                    BxsP�  �          @��H?�  �Tz��33��(�C��\?�  �.{�!G����C�n                                    Bxs_�  �          @���?�Q��\(�������C��\?�Q��8������
C�/\                                    BxsnJ  �          @�>�Q��p�׿˅����C��q>�Q��I���%���C�C�                                    Bxs|�  T          @��?u�L(���\�	��C�XR?u�=q�Fff�H(�C�Ф                                    Bxs��  �          @��?8Q��aG���p���=qC��=?8Q��4z��8���3�C�0�                                    Bxs�<  �          @��\?=p��XQ��	����
=C�&f?=p��(Q��A��?C���                                    Bxs��  �          @��H?Y���Tz��p��\)C�9�?Y���#33�C�
�C  C�E                                    Bxs��  �          @���?��R�l�Ϳ+��\)C�B�?��R�W
=�ٙ���
=C�#�                                    Bxs�.  �          @���?�\)�tz�\��Q�C�,�?�\)�c33���H��p�C�                                    Bxs��  �          @�z�@!��QG�>\)?�33C�� @!��J�H�O\)�3�C�K�                                    Bxs�z  �          @�z�@-p��G�=�G�?�G�C�w
@-p��AG��J=q�0��C��                                    Bxs�   �          @��@2�\�C�
>.{@�C�(�@2�\�>�R�8Q���
C��=                                    Bxs� �  T          @��
@$z��Mp�=��
?�ffC�W
@$z��Fff�Y���?�C�ٚ                                    Bxs�l  T          @���@�QG�=�\)?n{C���@�I���aG��JffC�T{                                    Bxs�  �          @�Q�?�
=�hQ�����C�h�?�
=�]p���=q�z�HC��                                    Bxs�,�  �          @~{?333�u>��@��C�*=?333�s33�8Q��)�C�9�                                    Bxs�;^  �          @�  ?�(��p�׾���n{C���?�(��a녿����C���                                    Bxs�J  �          @�G�?��\�w
=����
=C�^�?��\�b�\��\)��(�C�H                                    Bxs�X�  �          @��\?�\)�z=q>k�@K�C�H?�\)�s�
�k��Qp�C�7
                                    Bxs�gP  �          @{�?��\�q�=���?�Q�C��{?��\�h�ÿ�G��n�RC�ٚ                                    Bxs�u�  �          @|��?h���u��G����C���?h���h�ÿ�p���33C�H                                    Bxs���  �          @{�?c�
�s33>u@c33C��?c�
�mp��aG��PQ�C��R                                    Bxs��B  �          @{�?^�R�r�\>�@�
=C�}q?^�R�p�׿(���=qC���                                    Bxs���  �          @z=q?�G��p  =L��?8Q�C���?�G��fff����|(�C��{                                    Bxs���  �          @{�?��n{>�33@�(�C���?��j=q�B�\�3
=C��q                                    Bxs��4  �          @x��?u�n{=�\)?��\C�>�?u�dzῃ�
�y�C��f                                    Bxs���  T          @w
=?@  �p  ������C���?@  �\�ͿǮ��
=C�#�                                    Bxs�܀  �          @w
=?aG��l(��!G��\)C���?aG��U�ٙ���G�C�h�                                    Bxs��&  �          @u�?�=q�dz�L���B=qC�p�?�=q�J�H������z�C�n                                    Bxs���  �          @u�?����aG�������{C�<)?����O\)�����\C�
                                    Bxs�r  �          @u�?���[��Ǯ��G�C��=?���J=q�����ffC���                                    Bxs�  �          @u�?˅�\�;������C�Y�?˅�I����p���\)C�Z�                                    Bxs�%�  �          @w�?�Q��Tz�s33�dQ�C�w
?�Q��8�ÿ�33��ffC�&f                                    Bxs�4d  �          @z=q?�p��S�
������C��3?�p��4z���� z�C��                                    Bxs�C
  �          @y��?�33�Dz῵��
=C��q?�33� ���G��=qC��3                                    Bxs�Q�  T          @z=q?�\)�HQ쿱���\)C�t{?�\)�#�
�G���C��                                    Bxs�`V  �          @r�\?�=q�9�������33C�#�?�=q�33���HC�>�                                    Bxs�n�  �          @s�
?����!��	���
p�C��q?��ÿ�G��3�
�?�C���                                    Bxs�}�  �          @s�
?�Q��'���(����C�@ ?�Q��33�*�H�2G�C�Ǯ                                    Bxs��H  �          @s�
@
=�(�ÿٙ���Q�C�L�@
=�   ��H�Q�C�>�                                    Bxs���  �          @p��@�� �׿��
��G�C�j=@���
=�{�ffC�/\                                    Bxs���  �          @o\)@Q��,(������C�0�@Q��Q��
=q��HC�|)                                    Bxs��:  T          @vff?�ff�N�R���\�\)C���?�ff�0�׿��H���C��f                                    Bxs���  �          @qG�@��<(����
�~�\C��f@���R�����\)C�<)                                    Bxs�Ն  �          @q�@
=q�=p��n{�d��C�@
=q�!녿����ffC�8R                                    Bxs��,  �          @}p�@'��0  ��{��{C��{@'��녿�����HC���                                    Bxs���  �          @~�R@(���1G���=q�|��C��3@(���33�����p�C�s3                                    Bxs�x  �          @���@$z��;���  �fffC�� @$z���R�������C�                                      Bxs�  �          @�Q�@"�\�<(��n{�X(�C�h�@"�\� �׿�����C��=                                    Bxs��  �          @z=q@z��A녿Y���H��C��q@z��'��޸R�ӅC�Ǯ                                    Bxs�-j  �          @\)@!G��8�ÿ���x  C��\@!G���H��33��\C�
                                    Bxs�<  �          @vff@Fff�����˅��G�C�|)@Fff�c�
���H��=qC�f                                    Bxs�J�  �          @hQ�@AG���G��\���C�Ǯ@AG����H��  ��RC�j=                                    Bxs�Y\  �          @XQ�@)�����ÿ������C��3@)����������C�                                      Bxs�h  �          @g
=@=p�������\)C�.@=p��h�ÿ�����C�xR                                    Bxs�v�  �          @z=q@]p��u�\����C�:�@]p���G���  �ԏ\C�^�                                    Bxs��N  �          @w�@aG��Y����\)���C�7
@aG��\�Ǯ���RC��=                                    Bxs���  �          @X��@I���녿������
C���@I���L�Ϳ�Q�����C�'�                                    Bxs���  �          @<��@333��p��W
=��33C�>�@333��Q�k����C��                                    Bxs��@  �          @1G�=��*=q��\)����C�N=�� �׿fff��C�b�                                    Bxs���  �          @7
=�L���4z�>\)@1G�C���L���.{�B�\�x  C��                                    Bxs�Ό  �          @6ff=#�
�5���
���HC�\)=#�
�*�H�z�H��z�C�b�                                    Bxs��2  �          @(Q쾏\)�&ff<��
?�C����\)��R�L����{C��f                                    Bxs���  �          @0  �
=�(��>B�\@�33C��
�
=�$z�&ff�^{C��=                                    Bxs��~  �          @P  �����?#�
A6{Cd�)���p�����*�HCeǮ                                    Bxs�	$  �          @Vff�ff��
?
=qA��C`�=�ff�
=�aG��s�
Ca+�                                    Bxs��  �          @U�"�\�?   A	�C[� �"�\��þB�\�S�
C\+�                                    Bxs�&p  �          @QG��  ��>�@ffCb���  ��\�!G��1�Ca��                                    Bxs�5  �          @U���R�p�>L��@Z�HCcٚ��R�������%p�Cc
                                    Bxs�C�  �          @Mp������>��R@���Ce{���Q���	G�Cd�                                     Bxs�Rb  �          @R�\�G��ff>k�@\)Cb��G��33�
=q�
=Cac�                                    Bxs�a  �          @S33���G�>.{@>�RC_�����p�����HC^�R                                    Bxs�o�  �          @O\)�(�ÿ���>.{@B�\CW��(�ÿ���(���ffCVk�                                    Bxs�~T  �          @Q��.{��\>�p�@�=qCU!H�.{��ff�aG��s�
CU��                                    Bxs���  �          @N{�+���  >��@�ffCU0��+��޸R���
����CU\                                    Bxs���  �          @XQ��,�Ϳ�Q�?��A$��CW���,���녽u�p��CX�R                                    Bxs��F  �          @P���3�
��G����
��(�CP(��3�
����\�33CN��                                    Bxs���  �          @N�R�6ff��p�=�\)?��CO���6ff��
=������
CN�H                                    Bxs�ǒ  �          @K��2�\��33?�\A�CN�H�2�\���R<��
>�ffCP�                                    Bxs��8  �          @@  �0  �:�H?^�RA�Q�CB���0  �xQ�?
=A6ffCGp�                                    Bxs���  �          @C�
�.{�Ǯ?��AϮC<{�.{�L��?��A�
=CDk�                                    Bxs��  �          @L(��6ff���
?}p�A�(�CG���6ff���?��A,z�CLff                                    Bxs�*  �          @B�\�*�H�Y��?�
=A���CE���*�H��Q�?W
=A�{CK��                                    Bxs��  �          @1G���\�c�
?�(�A���CIJ=��\���R?\(�A��CPz�                                    Bxs�v  �          @�\���R��Q�=���@�RCd�����R�У׾��=�Cc�)                                    Bxs�.  �          @p��#�
�Q���_\)C�q�#�
��=q�����C�Z�                                    Bxs�<�  �          @2�\���
�   ��=q���C������
�   �����+
=C���                                    Bxs�Kh  �          @7
=�B�\�*�H�h�����C��;B�\�{�޸R�C���                                    Bxs�Z  �          @@  ���5��333�Z�\C�W
���(��˅��
C��)                                    Bxs�h�  �          @333���,(��+��_�C��f����
�\�Q�C�n                                    Bxs�wZ  �          @3�
���
�)���O\)���\C�'����
��R��33��C���                                    Bxs��   T          @333>aG��'��c�
���HC�XR>aG������H���C��3                                    Bxs���  �          @9��?}p��!G���G���  C���?}p���\��ff���C��f                                    Bxs��L  �          @<��?�{���33���\C�{?�{���ÿ�\)�C�aH                                    Bxs���  �          @5?�G���zΉ���ffC�Q�?�G���Q���G�C�g�                                    Bxs���  �          @#�
?У׿��Ϳ���C�7
?У׿J=q����2�RC��                                    Bxs��>  �          @�R?��H�����z���C��)?��H��Ϳ�Q��-(�C��                                    Bxs���  �          @1G�?�����Ϳ��
� �C�:�?����G��33�?�
C�(�                                    Bxs��  �          @Fff@�ÿs33�G��#33C��@�þW
=�{�7�
C�*=                                    Bxs��0  �          @=p�?��R�W
=���,�RC��3?��R�����(��>�
C��H                                    Bxs�	�  �          @!�?޸R�.{��Q��({C�j=?޸R�u���8p�C��                                    Bxs�|  �          @(Q�?�=q=#�
��(��L�R?�{?�=q?5��=q�:�A�33                                    Bxs�'"  �          @{?���>k���p��Uz�A�?���?aG����
�9ffA���                                    Bxs�5�  �          @��?�>W
=��z��Tz�A	��?�?Y����(��8�A�(�                                    Bxs�Dn  �          @3�
?���<��
��(��M�?\(�?���?0�׿��;��A�z�                                    Bxs�S  �          @n{@���*=q�xQ��yG�C�1�@���
�H������C�                                      Bxs�a�  �          @��@0  �)��������C�
=@0  ���H�
=�z�C�AH                                    Bxs�p`  �          @��R@*�H�8�ÿ��R��=qC�ff@*�H�������
��C�Ff                                    Bxs�  �          @��R@)���<�Ϳ�33��  C��R@)����\����RC���                                    Bxs���  �          @�  @5��5���33����C��H@5��
�H��\��HC�=q                                    Bxs��R  �          @��\@\)�U����H���C�` @\)�-p������\C�Ff                                    Bxs���  �          @�G�@Q��dzῃ�
�^�RC�^�@Q��@  ������C���                                    Bxs���  �          @��?���l�Ϳfff�HQ�C��\?���J=q�����RC��3                                    Bxs��D  �          @�{?�G��j�H�z�H�W�C�Ф?�G��G
=�(���=qC���                                    Bxs���  �          @�?�ff�e��{�w�C�K�?�ff�>�R��\�p�C���                                    Bxs��  �          @��
?�\)�tzῌ���g
=C��?�\)�L���
=��HC�                                      Bxs��6  �          @�{@�R�K���G��a�C��{@�R�(Q���
���C���                                    Bxs��  �          @��
@$z��@  ��33���C�K�@$z��=q������C�l�                                    Bxs��  �          @�p�@���N�R��ff�j=qC�Q�@���*=q������C�\                                    Bxs� (  �          @��@!G��U��xQ��R{C���@!G��1�����=qC��                                    Bxs�.�  �          @��@(Q��E����R��{C�AH@(Q�����\)��C���                                    Bxs�=t  �          @��@%��L(���=q�lQ�C��H@%��&ff�����C�c�                                    Bxs�L  �          @�Q�@{�S�
�xQ��S\)C�Y�@{�0  �z���Q�C��                                    Bxs�Z�  �          @�Q�@�R�Vff�L���.�HC�K�@�R�5��Q����C��\                                    Bxs�if  �          @�ff@%�P�׾����C�9�@%�8Q�˅���HC���                                    Bxs�x  �          @�z�@
=q�j=q��ff�]p�C�L�@
=q�C33������C��\                                    Bxs���  �          @��@p��dzῐ���qC�޸@p��<(�����C�z�                                    Bxs��X  T          @�\)@33�c�
�h���G�
C���@33�@  ����C�+�                                    Bxs���  �          @��\@Q��U�u�X��C�K�@Q��1������C��                                    Bxs���  T          @��@Q��^{����p(�C�Ǯ@Q��6ff�����C�e                                    Bxs��J  �          @�=q@   �l�Ϳp���IG�C�8R@   �G������C�\)                                    Bxs���  �          @��R@33�p�׿(����C���@33�QG����H����C��=                                    Bxs�ޖ  �          @�=q@�R�\�Ϳ�������C�s3@�R�/\)�{��HC��\                                    Bxs��<  �          @�\)@p��Vff�
=�C�.@p��:=q��G���{C�!H                                    Bxs���  �          @�Q�@ff�c33���
����C��@ff�S33��ff��Q�C��)                                    Bxs�
�  �          @��\@��g
=���
��
=C��=@��P  ��������C��                                    Bxs�.  �          @��@��hQ��
=���\C��@��N�R���H���HC�q                                    Bxs�'�  �          @�\)@��c33�������C�Y�@��L(�������\)C�Ǯ                                    Bxs�6z  �          @�{@ff�p  �k��:�HC��@ff�Z=q������Q�C�K�                                    Bxs�E   �          @�{@�p�׾�{��C��@�XQ��Q�����C�Y�                                    Bxs�S�  �          @���@��s�
��G����C���@��X�ÿ�ff��=qC�~�                                    Bxs�bl  �          @�\)?ٙ��u>��@�(�C��?ٙ��n�R���\�_33C�B�                                    Bxs�q  
�          @�?��n{>�\)@p��C��=?��c�
�����tQ�C�,�                                    Bxs��  �          @�=q@
=�]p�����
=C��\@
=�Mp���ff��G�C���                                    Bxs��^  �          @�{@�
�e�����C��f@�
�J=q��\����C��                                    Bxs��  �          @�?����l�Ϳ!G��	�C�  ?����Mp���Q���{C��                                    Bxs���  �          @�ff?��
�qG�����=qC���?��
�S�
��\)�ӅC��                                    Bxs��P  �          @�  @33�l�;���^�RC�}q@33�U��{��{C��                                     Bxs���  �          @���@��l�;�p����RC���@��S33���H��\)C�!H                                    Bxs�ל  �          @���@
=�^{?E�A&ffC�!H@
=�`  ����C��)                                    Bxs��B  �          @�{@{�aG�>W
=@;�C�!H@{�Vff��{�v�HC��=                                    Bxs���  �          @�=q@��^�R�\)��\C�ff@��K���
=���C��3                                    Bxs��  �          @��\@z��\��?�A�C�|)@z��Z�H�@  �*�HC��)                                    Bxs�4  �          @�33@���\��?�@�RC��f@���Y���L���333C�R                                    Bxs� �  �          @�(�@
�H�^�R>��@ϮC���@
�H�Y���^�R�B�\C�AH                                    Bxs�/�  
�          @�(�@Q��N{?uAXQ�C�7
@Q��Vff���R��(�C���                                    Bxs�>&  �          @��@$z��HQ�?�p�A���C���@$z��W
=�#�
�   C���                                    Bxs�L�  �          @�p�@(���@��?���A{\)C���@(���Mp���Q쿜(�C��
                                    Bxs�[r  T          @�@$z��J�H?G�A,��C���@$z��N�R�����G�C�AH                                    Bxs�j  �          @�p�@�R�S33?��AG�C�u�@�R�Q녿.{�\)C��f                                    Bxs�x�  �          @�z�@(Q��C�
?k�AL��C�N@(Q��K����R��G�C��                                    Bxs��d  �          @�{@333�:�H?��Ar=qC��f@333�G���Q쿡G�C��
                                    Bxs��
  �          @�@6ff�5?��Ar=qC��\@6ff�B�\��\)�uC��{                                    Bxs���  �          @s33@333�Q�?��\A���C�b�@333���>�p�@�(�C�ff                                    Bxs��V  �          @}p�@J=q�Q�?�{A���C�@J=q���>W
=@E�C�s3                                    Bxs���  �          @~{@H����R?k�AW33C�Z�@H���=q    =#�
C�E                                    Bxs�Т  �          @���@>�R�#�
?L��A8��C���@>�R�*�H�u�W�C�\                                    Bxs��H  �          @~{@A��{?(��Ap�C�c�@A��"�\��{��33C�f                                    Bxs���  
�          @�{@>�R�:�H>��@33C��=@>�R�0  �z�H�Y��C��H                                    Bxs���  �          @�33@.�R�E>W
=@<(�C���@.�R�<(��}p��^=qC�u�                                    Bxs�:  �          @�z�@-p��I��=�?�
=C�T{@-p��<�Ϳ����u��C�>�                                    Bxs��  �          @�\)@%�S�
<�>ǮC��)@%�Dzῠ  ��z�C�\                                    Bxs�(�  �          @��H?���g�>L��@5�C���?���Z=q���H��{C�j=                                    Bxs�7,  �          @}p�?�ff�`  >\)?�p�C��3?�ff�Q녿�p���p�C�XR                                    Bxs�E�  T          @���@p��c33�Tz��4(�C��@p��<���	����p�C�c�                                    Bxs�Tx  �          @\)@��1녿�33���\C��{@��G�����C�&f                                    Bxs�c  �          @��@4z���ÿ޸R����C��)@4z���
�!���
C��f                                    Bxs�q�  �          @���@-p��(Q�޸R���
C���@-p��޸R�'���RC��
                                    Bxs��j  �          @~{@�R�p����H��C�%@�R���
�*=q�.��C�g�                                    Bxs��  
�          @a�@�R?#�
����,=qAh  @�R?\�����ffA���                                    Bxs���  �          @c33@ff?������*��B=q@ff@
=��{��=qBAz�                                    Bxs��\  �          @|(�@G�?�Q��#�
�"�B!�R@G�@2�\�������BK(�                                    Bxs��  �          @~�R@�\@	������
=B,�H@�\@<(���z���{BPz�                                    Bxs�ɨ  �          @|(�@
=q@(��\)�\)B5��@
=q@?\)��
=���BX�                                    Bxs��N  �          @r�\@!�?У��33�Q�B�@!�@����(����\B.
=                                    Bxs���  �          @s33@�R@z��G��  B+�
@�R@333���\��=qBN{                                    Bxs���  �          @vff?Ǯ@"�\�(��\)Bi�R?Ǯ@R�\���R��(�B�\)                                    Bxs�@  �          @p��?�33@#33�Q���
Bt�?�33@R�\����G�B�Ǯ                                    Bxs��  �          @p  ?�p�@ ���ff��Bm��?�p�@N�R��z�����B��q                                    Bxs�!�  �          @l��?��R@!��  �Bm��?��R@Mp��������B�=q                                    Bxs�02  �          @n{?�z�@���H�"��Bo�R?�z�@L�Ϳ�  ���B�k�                                    Bxs�>�  �          @tz�?\@�&ff�+��Bc�?\@K��������HB���                                    Bxs�M~  �          @q�?���@Q��p��"�RB`\)?���@J�H�����{B|                                    Bxs�\$  �          @z�H?��@p�������BR�
?��@Mp����H���
Bn��                                    Bxs�j�  �          @��?���@&ff�   �ffBT�?���@XQ쿠  ��{Bp33                                    Bxs�yp  �          @�33?��H@C33��G�Beff?��H@n�R�k��DQ�Byff                                    Bxs��  �          @���?���@0  �,(��G�B`�?���@fff��{��z�B{
=                                    Bxs���  �          @�33@   @'
=��H�Q�BR{@   @W
=��z���ffBm                                      Bxs��b  �          @���@(�@   �\)��BC{@(�@R�\���
���Ba�                                    Bxs��  �          @\)@�@   �(����BC�\@�@J�H��  �h��B]��                                    Bxs�®  �          @���@{@'
=��
��=qBFQ�@{@N{�Tz��?�B]��                                    Bxs��T  �          @���@  @#�
�Q�� �BB��@  @L�Ϳh���Q�B[�\                                    Bxs���  �          @���@ff@.{�G��(�BDz�@ff@Y���xQ��S�B]p�                                    Bxs��  �          @tz�?У�@G���Q���=qByff?У�@\(��L�Ϳ@  B�#�                                    Bxs��F  �          @z�H?�@A녿������HBt?�@aG���ff��z�B�Q�                                    Bxs��  
�          @^�R@�?�p���G���G�B)\)@�@!녿L���W�
BE(�                                    Bxs��  �          @^�R@\)?��H���
����B$�
@\)@!G��Q��[�BA\)                                    Bxs�)8  �          @`��@,(�?�=q�����(�A��@,(�@��B�\�I�B��                                    Bxs�7�  �          @j=q@8��?��ͿǮ�ə�A�@8��@ff�B�\�@��B��                                    Bxs�F�  �          @z�H@HQ�?�G������=qA�Q�@HQ�@���G��pz�B33                                    Bxs�U*  �          @x��@Dz�?�p�����{A�p�@Dz�@��ff�{\)B	                                      Bxs�c�  �          @u@-p�@ �׿���  B�H@-p�@$z�Tz��G\)B-��                                    Bxs�rv  �          @qG�@1G�?�׿�Q���z�B��@1G�@=q�B�\�<Q�B$=q                                    Bxs��  �          @N{@Q�?�33���H��Q�A�@Q�?�\)�@  �]B�                                    Bxs���  �          @J�H?���?�����/p�B?���@��Q��ڏ\BA��                                    Bxs��h  �          @R�\?޸R?�\)�  �-B+ff?޸R@��������G�BX=q                                    Bxs��  T          @J=q?�\?�Q��33�<�B33?�\@�\������\)BD��                                    Bxs���  
�          @H��?�?�
=����{B+��?�@zῂ�\���
BP�                                    Bxs��Z  �          @B�\?���?���\)�=qB3�?���@z�8Q��\  BO�R                                    Bxs��   �          @>{?�?޸R��Q����B8=q?�@녿O\)�~�HBV�                                    Bxs��  �          @HQ�?�G�@z���
����BG  ?�G�@!G�����\B\�                                    Bxs��L  �          @@  ?�33@녿����  Bj33?�33@'��aG���ffBw�\                                    Bxs��  �          @<(�?�=q@!녿��5�Bx��?�=q@!�?
=qA*�RBy33                                    Bxs��  �          @R�\?�@   �aG��{
=BR
=?�@(��>�  @���BX
=                                    Bxs�">  �          @Y��@J�H?fff>�33@�  A~�H@J�H?.{?0��AA��A@Q�                                    Bxs�0�  �          @\��@R�\�W
=�#�
�\)C��{@R�\�B�\��Q��ÅC��                                     Bxs�?�  �          @^�R@1G��G��#�
�(Q�C��f@1G���\��  ����C���                                    Bxs�N0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxs�\�   =          @*=q@ff�У׾���U�C�q@ff��z�Tz���
=C��                                    Bxs�k|  �          @Tz�@#33�
=�.{�9��C�,�@#33����ff��33C��                                    Bxs�z"  �          @k�@.�R����G����
C�+�@.�R�	��������ffC��H                                    Bxs���  �          @^{@(����>��@��
C�y�@(����L���Up�C�1�                                    Bxs��n  �          @�@@  �9��>�z�@~�RC�  @@  �/\)�xQ��V�\C��                                    Bxs��  �          @���@E��HQ�=���?��\C�H�@E��7���  ���C���                                    Bxs���  �          @��H@p��dz�B�\�\)C�K�@p��I����Q����C��                                    Bxs��`  �          @�ff@3�
�Fff=��
?�{C��@3�
�5��  ��33C�Y�                                    Bxs��  �          @|(�@@�����>��@p��C�t{@@���z�Q��AC�1�                                    Bxs��  �          @���@S�
�{>�Q�@�33C�R@S�
�
=q�!G����C�w
                                    Bxs��R  �          @|(�@P�׿��R?B�\AEC��{@P�׿���>#�
@!�C�                                    Bxs���  �          @�(�@L��?Q�@��B�RAe�@L�;�z�@ ��B(�C�l�                                    Bxs��  �          @��
@o\)�L��?�p�A�z�C��H@o\)�J=q?��A���C��                                    Bxs�D  �          @^�R@G
=��\)?�(�AɅC�b�@G
=�c�
?���A�33C��                                    Bxs�)�  �          @z�?ٙ�>��?�\)B
=@�=q?ٙ���33?�=qA�(�C�:�                                    Bxs�8�  �          @?\)@��?   ?��HB=qA<  @����z�?�  B��C��R                                    Bxs�G6  T          @7
=@(�?&ff?�(�B��A��@(��#�
?�=qB{C��f                                    Bxs�U�  �          @0��@p�?�R?��
B��Az�R@p�����?��B�\C���                                    Bxs�d�  �          @#33@��>�?�G�A�Q�@Y��@�;�(�?�Q�A�=qC�xR                                    Bxs�s(  �          @�\)@P  ��\?(��A��C�k�@P  �ff�����G�C��                                    Bxs���  �          @�
=@mp��ff?8Q�AC��{@mp�����p�����C�c�                                    Bxs��t  �          @�\)@�Q��\)?z�@�C��H@�Q��Q쾔z��o\)C�q                                    Bxs��  �          @�p�@y������?!G�A�RC���@y���녾�z��o\)C�E                                    Bxs���  �          @��\@^{���H?�\@��C�H�@^{���R������ffC�                                      Bxs��f  �          @x��?�G��P�׿�=q���\C�.?�G��������  C�Ф                                    Bxs��  �          @j=q@p��)���Tz��XQ�C���@p��G���z��  C���                                    Bxs�ٲ  �          @g
=@�
�'
=������C�#�@�
��G��ff�#��C��H                                    Bxs��X  �          @[�?�ff�\)��=q�Q�C�ff?�ff�����(Q��JG�C��                                    Bxs���  �          @tz�?�(��+����HC�aH?�(����R�C33�U(�C��)                                    Bxs��  �          @u?�
=�7���ff��\)C�33?�
=���
�8Q��Fp�C��\                                    Bxs�J  �          @~{?�33�.{�!G��{C��q?�33�����]p��s��C�E                                    Bxs�"�  �          @�p�?aG��-p��Z=q�HG�C���?aG��n{��  ��C���                                    Bxs�1�  �          @��R?h���5��Tz��@��C��?h�ÿ�����\)B�C�E                                    Bxs�@<  �          @��?Y���H���E��.
=C��
?Y����Q�����{C�AH                                    Bxs�N�  �          @�
=?u�?\)�K��5�C���?u���\��W
C��H                                    Bxs�]�  �          @���?�Q��
=q�_\)�Z��C�t{?�Q�\���H�fC�R                                    Bxs�l.  �          @��\?�
=���K��P33C�Ff?�
=���s�
\)C��R                                    Bxs�z�  �          @|(�?����{�@���J33C�q?��Ϳ&ff�l(��C��3                                   Bxs��z  �          @�G�?#�
���XQ��e  C���?#�
��33�}p�#�C���                                   Bxs��   �          @�\)?����R�>{�:�HC��3?���aG��qG�u�C��q                                    Bxs���  �          @�
=@(���HQ�>�
=@��C�R@(���>�R���
�g\)C���                                    Bxs��l  T          @��@:�H�1G�?G�A/\)C�H�@:�H�4z�
=q���
C�                                      Bxs��  �          @�Q�@?\)�>�R�L���.�RC���@?\)�%��\��=qC��                                    Bxs�Ҹ  �          @�G�@QG��%�n{�H(�C���@QG���33���R��(�C���                                    Bxs��^  �          @��R@B�\�0  �J=q�/\)C���@B�\�ff������Q�C��                                    Bxs��  �          @��@'
=�2�\�����=qC��f@'
=����<(��.�C��f                                    Bxs���  �          @���@���0  �
=��C��\@�Ϳ���Vff�G�
C�.                                    Bxs�P  �          @�33@!��(Q�����(�C��3@!녿�G��Tz��Fz�C���                                    Bxs��  �          @�@.{����<(��3�HC���@.{>��
�H���C��@�                                    Bxs�*�  �          @�  ?���=q�s33�f�\C�q�?�>��R����3A�R                                    Bxs�9B  �          @���?�p�����E�D��C���?�p��L���dz��sQ�C�/\                                    Bxs�G�  �          @��@Dz��:=q�k��C�C�C�@Dz���������HC�U�                                    Bxs�V�  T          @�\)@A��AG�������C��{@A��33�'
=���C��                                    Bxs�e4  �          @�Q�@=p��?\)��{��{C�Y�@=p���
=�333���C�w
                                    Bxs�s�  �          @�Q�@Mp��.�R�˅��\)C���@Mp��ٙ��*=q��C�
=                                    Bxs���  �          @�G�@XQ��{��p���33C��=@XQ쿴z��*=q�{C���                                    Bxs��&  �          @��R@N�R�$z��z����HC�� @N�R���
�)����C�W
                                    Bxs���  �          @���@g
=�
=�ٙ�����C���@g
=�����{��HC���                                    Bxs��r  �          @�Q�@p�׿����G�����C�L�@p�׿G�������RC�%                                    Bxs��  T          @�{@a녿�{��33��{C��@a녿G��"�\���C���                                    Bxs�˾  �          @��@fff��33�(Q��(�C�'�@fff>B�\�7
=��@<��                                    Bxs��d  �          @�z�@Q녿^�R�J=q�+�C��@Q�?.{�Mp��.�A:ff                                    Bxs��
  �          @��H@>{� ���ff����C�� @>{���\�AG��,p�C�u�                                    Bxs���  �          @��R@,���TzῙ���{�C��\@,������&ff��C�<)                                    Bxs�V  �          @��@/\)�R�\��=q����C��@/\)�(��:�H��HC���                                    Bxs��  �          @���@@  ��R�{��C�1�@@  ��
=�G��033C�K�                                    Bxs�#�  �          @��@5��-p�����ͮC��@5����
�;��)=qC�Ǯ                                    Bxs�2H  �          @�Q�@S33�(������G�C��R@S33����333�ffC�H�                                    Bxs�@�  �          @���@`  �$z��G����C��)@`  ����>�R�C���                                    Bxs�O�  �          @��@_\)�{�p���ffC�W
@_\)��z��Fff� ��C��                                    Bxs�^:  �          @�=q@U�)���\)�݅C��f@U����Mp��'z�C�O\                                    Bxs�l�  �          @�@^�R�*=q�G��ۮC�S3@^�R����P  �$��C�Ф                                    Bxs�{�  T          @��
@A����C�
���C��@A녾�p��j=q�H�C�~�                                    Bxs��,  �          @��\@N�R��Q��R�\�+�HC���@N�R>���e�?��@��                                    Bxs���  �          @��@P  ��p��Tz��.�
C���@P  >��`���;��A��                                    Bxs��x  �          @�G�@O\)����W��2�C���@O\)?#�
�^{�9��A2ff                                    Bxs��  �          @���@P  �Q��dz��;33C��f@P  ?s33�b�\�9(�A���                                    Bxs���  �          @�
=@Z�H�aG��`  �3(�C��@Z�H?^�R�`���3Q�Ac�                                    Bxs��j  �          @�
=@hQ�5�QG��%C�y�@hQ�?h���N{�"�RA`Q�                                    Bxs��  �          @�\)@N�R�:�H�l���@�C���@N�R?����g
=�:z�A�                                    Bxs��  �          @�ff@J�H���qG��F��C��=@J�H?�\)�b�\�6�
A��                                    Bxs��\  �          @���@E���G��|(��N��C��@E�?��H�k��<��A�G�                                    Bxs�  �          @���@`��?
=q�Vff�-z�A�@`��@�
�,���p�A��H                                    Bxs��  �          @�(�@aG�?���U�,��Aff@aG�@�
�+��ffA���                                    Bxs�+N  �          @��\@\(�?.{�S33�,�A3�@\(�@
=q�%��
B p�                                    Bxs�9�  �          @�{@\��?fff�C33�"33AiG�@\��@�R�����RB��                                    Bxs�H�  �          @�(�@Vff?u�C33�$�
A�@Vff@�\��R��RB	\)                                    Bxs�W@  �          @w�@1�?xQ��(Q��&��A�(�@1�@ff������{BQ�                                    Bxs�e�  �          @s33��  ?=p��j�H�fBؽq��  @���8Q��HQ�B�                                    Bxs�t�  �          @hQ�?�=q?�  �Tz�=qB+33?�=q@���(��)�B��{                                    Bxs��2  �          @j�H?�
=?���Dz��g�
B(G�?�
=@%���HBt�R                                    Bxs���  
�          @p  ?�Q�?����/\)�>�\B��?�Q�@,(���z�����BX�\                                    Bxs��~  �          @e�?�(�?�G��%��8�\Bz�?�(�@#�
��ff�ͅBQQ�                                    Bxs��$  �          @S�
?�\)?�{��5�B�?�\)@�
��z���ffBK�
                                    Bxs���  �          @N{?��R?�Q����!{B?��R@�׿�����
BB��                                    Bxs��p  �          @�  @(�?��� �����B{@(�@6ff������BEp�                                    Bxs��  �          @�G�@  @�R�*=q�(�B?(�@  @\�Ϳ�33�zffBc�                                    Bxs��  �          @_\)?���(��33��C��3?����z��>{�y
=C��)                                    Bxs��b  T          @`��?�G��5���z���
=C��?�G���Q��3�
�\{C�k�                                    Bxs�  �          @q�?(��+��"�\�+
=C�j=?(������`���
C�.                                    Bxs��  �          @r�\?5�-p�����"33C�^�?5��p��Z=q��C��                                    Bxs�$T  �          @}p�?xQ��(��:=q�?�
C��
?xQ�=p��n{ǮC�\)                                    Bxs�2�  �          @��?��
�ٙ��dz��lz�C�y�?��
>aG��|��� A=q                                    Bxs�A�  |          @�z�?��\)�s�
�|�
C��{?�?����g
=�f�RB�H                                    Bxs�PF  �          @�  @�?�(��\(��J33B0�\@�@X����
��z�Bl33                                    Bxs�^�  �          @�z�?�z�@�K��:{BJ�?�z�@e���z����Bw                                    Bxs�m�  �          @��?�\@8Q��2�\��Bj
=?�\@w���ff�]p�B���                                    Bxs�|8  �          @���?޸R@U�z���=qBy��?޸R@z�H�B�\�%�B�.                                    Bxs���  �          @�
=?�(�?���hQ��lp�B^�?�(�@U���
�33B��{                                    Bxs���  �          @��?��
?�  �^�R�hz�BW33?��
@N{������B���                                    Bxs��*  �          @xQ�?333?�
=�Z=q�wz�B���?333@HQ��
=q�{B��
                                    Bxs���  �          @k�?k�@�
�+��<�HB���?k�@Tz῜(���z�B��                                    Bxs��v  �          @j�H>Ǯ@{�9���P�
B�L�>Ǯ@Vff���H��
=B��H                                    Bxs��  �          @{�?#�
@   �<���Cz�B�=q?#�
@g���{����B��                                    Bxs���  �          @e?fff@��,(��B��B�#�?fff@Mp����
��{B��3                                    Bxs��h  �          @b�\@z�?�{�(���HB'p�@z�@,(���G���(�BQ�\                                    Bxs�   T          @p��?�R@&ff�!��-�B�?�R@`�׿p���lQ�B���                                    Bxs��  
�          @�z�>�@G
=�+��!�\B���>�@��׿Q��6{B��                                     Bxs�Z  �          @�  >�ff@B�\�:=q�-�HB�Q�>�ff@��\��ff�fffB��                                    Bxs�,   �          @��?�  @;��8Q��,  B�B�?�  @}p���=q�m�B��                                    Bxs�:�  �          @�
=?xQ�@;��7
=�+\)B��\?xQ�@}p�����h��B���                                    Bxs�IL  �          @��H?�G�@6ff�%��!��B�
=?�G�@p  �Y���E��B��f                                    Bxs�W�  �          @w�@,��?�  ����A�  @,��@녿�G�����B ��                                    Bxs�f�  �          @tz�?��
@
=�'
=�-ffBG(�?��
@G����R���Bp��                                    Bxs�u>  �          @vff@*�H?���33��A��@*�H@33�����=qB#                                      Bxs���  �          @x��?��H@!��%��*{B�?��H@^{�}p��r{B�p�                                    Bxs���  �          @\)>Ǯ@Mp��z���
B�B�>Ǯ@{���G����HB�Ǯ                                    Bxs��0  �          @\)?
=@J=q�z��G�B�Ǯ?
=@y��������B��R                                    Bxs���  �          @��333@:�H�:�H�0�HB��)�333@~�R�����s33B��)                                    Bxs��|  �          @�G��
=@�N�R�T�B�\)�
=@hQ��33��p�Bƀ                                     Bxs��"  
�          @w����@ ���N{�f
=B��H���@U���ᙚB��H                                    Bxs���  �          @l(��.{?����W
=  B��f�.{@7
=��R��B��R                                    Bxs��n  �          @x�þL��?ٙ��[��}B�G��L��@J�H�Q��\)B�#�                                    Bxs��  �          @Z�H?!G����0���]  C��?!G��L���R�\aHC��                                    Bxs��  �          @Z�H>��H�p���N�R��C��\>��H?J=q�QG�L�Bh{                                    Bxs�`  �          @p�>��<#�
�ff§aH?Ǯ>��?���\�mB�
=                                    Bxs�%  �          @z�>B�\�B�\��
=¤�C�q�>B�\?L�Ϳ�\=qB��H                                    Bxs�3�  �          @�>Ǯ�^�R?�p�A�  C�5�>Ǯ��  �#�
�B�\C�˅                                    Bxs�BR  �          @��\?8Q��~�R?�  A�(�C�  ?8Q���ff�0����RC�޸                                   Bxs�P�  �          @�Q�?�Q���  ?O\)A0Q�C�H�?�Q��w������G�C���                                   Bxs�_�  �          @�{?�=q��Q�>�(�@�\)C���?�=q�l(���\)���C�%                                    Bxs�nD  �          @���?(����p�?��@�(�C��?(���x�ÿ�{����C��R                                    Bxs�|�  �          @��
>����\>��
@��RC�(�>��z=q����̣�C�~�                                    Bxs���  T          @��?�����=L��?.{C���?��n{����C��                                    Bxs��6  �          @��\?�\�����#�
�z�C�h�?�\�l���(����C��{                                    Bxs���  �          @�p�>�p���33�
=q��33C�o\>�p��`  �)����C��                                    Bxs���  �          @��>W
=���fff�B=qC�w
>W
=�J�H�8Q��(�RC��                                    Bxs��(  �          @�{?5��녿
=�33C���?5�N{�#33��C�5�                                    Bxs���  �          @��
?=p����׾Ǯ��C�:�?=p��Q��ff���C�^�                                    Bxs��t  �          @��H?!G��z�H�k��PQ�C���?!G��:�H�0���*��C�{                                    Bxs��  �          @~{?
=q�e������HC�G�?
=q�=q�@���J\)C�T{                                    Bxs� �  �          @u�z�H�Dz������p�C|J=�z�H���N�R�l33Co��                                    Bxs�f  �          @o\)�E��Q��/\)�>z�C|.�E��.{�dz�
=C]xR                                    Bxs�  �          @u��   ��\�R�\�s\)C~W
�   >��o\)¤\)C%�                                    Bxs�,�  T          @�(�����<���,(��"z�Cz&f�����Q��s33�Cd(�                                    Bxs�;X  �          @tz��=q�*�H��p���
Co^���=q����C�
�`\)C[��                                    Bxs�I�  �          @xQ��(��
=�ٙ�����C`��(���
=�*=q�1�CM�                                     Bxs�X�  �          @u���R��=q�5��@G�CZY����R=�\)�O\)�iz�C1�R                                    Bxs�gJ  �          @qG���H�fff�/\)�:�CHxR��H?���4z��A�
C%�R                                    Bxs�u�  �          @o\)�(��B�\�$z��4�
CEQ��(�?&ff�&ff�7��C%!H                                    Bxs���  �          @\)�l�ͿY��������(�C@�q�l�;����H����C5��                                    Bxs��<  �          @������׿=p��333�(�C>aH���׾�\)�z�H�Z�\C7�3                                    Bxs���  �          @�����
=��Ϳ\)��Q�C;aH��
=�B�\�B�\�#\)C6��                                    Bxs���  �          @�  ������G����
C;#����Ǯ��Q���{C9c�                                    Bxs��.  �          @hQ쾞�R�8Q��\){CQ𤾞�R?L�Ϳٙ��}\)B�{                                    Bxs���  �          @��?����5���\�|(�C�
=?���?�(��w��g�B�R                                    Bxs��z  T          @�G�?��k��c�
�u�
C���?�?�\)�K��N��B (�                                    Bxs��   �          @S33��Q�>��R�%�qC'� ��Q�?˅�z��0(�C�                                    Bxs���  �          @n�R�h�ýu<�?�C4���h�ý�\)    �#�
C5�                                    Bxs�l  �          @a��^�R�L�;k��q�C7O\�^�R�u������{C4�                                    Bxs�  �          @dz��L��?����Q��ffC�L��?�?
=AQ�C޸                                    Bxs�%�  �          @s33�I��?�z�G��@(�C�=�I��@�\>���@��\C\                                    Bxs�4^  �          @S�
��\)��\)�K�©� Cn��\)?����8Q���B��                                    Bxs�C  �          @g
=��z�=�Q��XQ�C/^���z�?��7
=�T33B�G�                                    Bxs�Q�  �          @vff��Q�?�\)�XQ��v�HC쿘Q�@:=q����
=B��                                    Bxs�`P  �          @l�;u?ٙ��L(��w{B�#׾u@Fff��\)����B��f                                    Bxs�n�  �          @p  ���@��1G��A�\B�\)���@R�\��G���\)B�Q�                                    Bxs�}�  �          @[���\)@(����ffB�
=��\)@:�H��R�,��B�G�                                    Bxs��B  �          @p����\@  �p����C8R��\@Dz�=p��5B�\)                                    Bxs���  �          @h�ÿ�Q�@&ff��  ��B����Q�@HQ�.{�,(�B�                                    Bxs���  �          @s33�����
=�>{�Q�Cc� ���=��
�Z=qB�C1                                    Bxs��4  �          @vff�������N{�p=qCa쿧�?��\(�L�C                                      Bxs���  �          @�\)�+����H��R�Q�C�\)�+��K��(Q����C�{                                    Bxs�Հ  �          @�z�\)������H��33C�#׿\)�<(��K��;
=C��3                                    Bxs��&  �          @����R���?�@�Q�C����R�y����p���G�C��H                                    Bxs���  �          @�
=>8Q���G�?.{A��C�K�>8Q��tz��  ��p�C�^�                                    Bxs�r  �          @�G�>�Q���
=>��H@�G�C�q�>�Q��w
=��G����C��=                                    Bxs�  �          @�  ?#�
���H?\(�A;�C�y�?#�
�|�Ϳ�����C���                                    Bxs��  �          @�G�?�\�,��@��BffC���?�\�c33?(��A\)C�J=                                    Bxs�-d  �          @��?�Q��\)@5�B.�C�z�?�Q��XQ�?��\A�  C��{                                    Bxs�<
  �          @��H?�33�x��?��AG�C�|)?�33�h�ÿ�  ��Q�C�/\                                    Bxs�J�  �          @�{?����p�>��
@�C���?���n�R�����  C��                                    Bxs�YV  
�          @�  >�Q���z�?h��A:ffC�` >�Q���
=���R��G�C�xR                                    Bxs�g�  �          @�
=?�����Q�?�=qAh��C�� ?����~�R��33�yC���                                    Bxs�v�  �          @��R?������L���&{C��q?���H���5�=qC�O\                                    Bxs��H  �          @�\)?��\��(�����c
=C���?��\�>{�E��.��C���                                    Bxs���  �          @�ff?�����
�@  �(�C��q?���H���1G���\C���                                    Bxs���  �          @�{?��H�~�R�u�I��C�� ?��H�Q��G���ffC��f                                    Bxs��:  �          @��
@�H�g
=>\@��C���@�H�Q녿�ff��ffC�:�                                    Bxs���  �          @��H@�fff?�\)Ap��C��@�j=q�h���D(�C���                                    Bxs�Ά  �          @�=q?�(��}p��B�\�$z�C�1�?�(��R�\�{��\)C��                                    Bxs��,  �          @w�?�z��L(���z���(�C��q?�z��{�Dz��Y�C��{                                    Bxs���  �          @y��?�p��O\)�ٙ��љ�C�e?�p�����H���YC���                                    Bxs��x  �          @�z�?�{�Tz��{��C��=?�{��\)�Tz��Uz�C�Z�                                    Bxs�	  �          @~{@z��/\)�����  C�&f@z��z��!G��%��C�,�                                    Bxs��  T          @�@a��ff�E��*ffC�� @a녿�녿�G���z�C�B�                                    Bxs�&j  �          @�Q�@X������n{�IG�C�W
@X�ÿ�����\��C���                                    Bxs�5  �          @���@Mp��=q������
=C��=@Mp��������RC��=                                    Bxs�C�  �          @���@HQ��(����
��ffC�
=@HQ쿣�
�%���C��)                                    Bxs�R\  �          @�  @8���!G����
��G�C�u�@8�ÿ�(��5�(ffC���                                    Bxs�a  �          @��@�
�#�
�(��ffC��@�
�h���Z�H�XQ�C�H�                                    Bxs�o�  �          @s�
?�{�  �*�H�0  C��f?�{���\�����C�H                                    Bxs�~N  �          @o\)@C33��{@�B\)C��
@C33��\)?˅A�C��                                     Bxs���  �          @��@o\)�L��?�G�AŮC���@o\)��{?�=qAmp�C�^�                                    Bxs���  �          @���@Z=q���@�\A�Q�C���@Z=q���?���A�\)C��                                     Bxs��@  �          @��@����333?�{A�{C�]q@�����Q�?�  AG\)C��q                                    Bxs���  �          @�G�@��H����?��A��C��@��H���?Q�A�HC�                                    Bxs�ǌ  �          @�z�@�
=����?���A�ffC���@�
=���?=p�Az�C��                                    Bxs��2  
�          @�{@��R���
?�33AV�RC��@��R��=#�
>�C�K�                                    Bxs���  �          @�@���ff?���A�C�ٚ@���   =L��?�RC��                                    Bxs��~  �          @�ff@����33?+�@���C��f@����ff�   ���
C�C�                                    Bxs�$  �          @�p�@�����R?Q�A33C�>�@�����������C��
                                    Bxs��  �          @�z�@�G�� ��?�A]��C�l�@�G����u�@  C�                                    Bxs�p  �          @��R@����33?&ff@���C���@����33�(����\)C��\                                    Bxs�.  �          @�z�@�G��.{�u�=p�C��@�G��녿�p����
C�J=                                    Bxs�<�  �          @��\@h���,(��8Q��(�C���@h�ÿ�z������
C�&f                                    Bxs�Kb  �          @�33@6ff�HQ�O\)�-C�+�@6ff��R��\)C��
                                    Bxs�Z  �          @�ff@1��\�;�z��s33C�k�@1��1��z����C��                                     Bxs�h�  �          @��
@:=q�N�R�������C��@:=q�!���\��33C���                                    Bxs�wT  �          @�(�@>�R�J�H����
=C���@>�R�p���
���\C�<)                                    Bxs���  T          @���@?\)�C33��Q쿚�HC�33@?\)�"�\��
=���
C��=                                    Bxs���  �          @tz�?�{�-p�?8Q�AUp�C�Y�?�{�+��Q��s33C�xR                                    Bxs��F  �          @�(�<��
�\���G����C�"�<��
��p��p  � C�C�                                    Bxs���  �          @�=q>����P���Q���C��>��Ϳ�G��o\)� C�o\                                    Bxs���  �          @qG�?
=q�5�����C�e?
=q����P����C�33                                    Bxs��8  �          @w�?�R�Mp����R���RC�� ?�R���XQ��x�RC�9�                                    Bxs���  �          @dz�333�p���S�
��Cik��333?s33�S�
ǮB�                                    Bxs��  �          @c33�B�\=�Q��\��¬�Cٚ�B�\?�33�8Q��a�
B��{                                    Bxs��*  �          @g�?=p��z�H�L(���C��?=p�?\(��N{\)BD�                                    Bxs�	�  �          @dz�?�{����G��t�C�S3?�{?E��L����RA�Q�                                    Bxs�v  �          @c�
?�  �
=�A��jC��\?�  ?����8Q��Yz�B�                                    Bxs�'  �          @l(�?У׿�{�G
=�fz�C��f?У�?5�N�R�t��A�
=                                    Bxs�5�  �          @r�\?��H���H�C�
�W{C���?��H>Ǯ�W��z33AK�                                    Bxs�Dh  �          @o\)?�(���  �;��P{C�}q?�(�>�\)�QG��w��A�\                                    Bxs�S  T          @mp�?�{�aG��I���lC���?�{?�p��2�\�E��B�
                                    Bxs�a�  �          @j�H?&ff��Q��P��(�C��=?&ff?��<���xz�B��=                                    Bxs�pZ  �          @vff?녿Ǯ�]p���C�H?�?�\�p��  B(p�                                    Bxs�   �          @~{?&ff�޸R�Vff�t
=C�=q?&ff>����p��p�A�33                                    Bxs���  �          @��ÿ}p��p���r�\ffC_��}p�?��H�mp�B�C33                                    Bxs��L  �          @�  �J=q��\)�b�\�|�Ct��J=q?   �w
=�RCaH                                    Bxs���  �          @y������,(��	����Cx𤿅�����P  8RCcǮ                                    Bxs���  �          @|�Ϳ��\�i���:�H�.�RC~p����\�-p��#�
�&  Cyh�                                    Bxs��>  �          @z�H<��
�1��.�R�1��C�:�<��
�fff�r�\�C��3                                    Bxs���  �          @~{>��H�$z��<���A��C�` >��H�(��w
={C�G�                                    Bxs��  �          @w�?Tz��;�����Q�C��?Tzΰ�
�^{�=C�k�                                    Bxs��0  T          @}p��+��&ff�l��C`(��+�?��^�R�B�u�                                    Bxs��  �          @������˅�l��Q�C��)��?�R�~�R¢B���                                    Bxs�|  �          @��H?���(��S�
�^G�C���?����Q��~{¢�3C��                                     Bxs� "  �          @�
=>��R� ���e��q�RC�u�>��R>L����33©�=B
=                                    Bxs�.�  �          @��þ.{�޸R�q�k�C�B��.{?
=q��(�¤k�B�\)                                    Bxs�=n  �          @�=q?L���`  �.�R�{C�w
?L�Ϳ�G���u�C��                                    Bxs�L  �          @�?O\)�Z�H�,�����C���?O\)���H����3C�}q                                    Bxs�Z�  �          @���?Tz��P  �+��z�C�  ?TzῪ=q��  Q�C��3                                    Bxs�i`  �          @�=q?����p���G��C��3?�����(��{��nG�C���                                    Bxs�x  �          @��
?z�H�k��
=��C�|)?z�H���R�p  �m33C�"�                                    Bxs���  �          @��?��
��Q������C��?��
�)���q��T�C��                                     Bxs��R  �          @�ff?������\�\���RC��=?����Dz��n�R�B��C�
                                    Bxs���  T          @��H?z�H��\)����  C�/\?z�H�A��e�A�C���                                    Bxs���  �          @�=q?
=��{�����C���?
=�*�H�g��S�HC�7
                                    Bxs��D  �          @�G�?=p���(��Y���-�C��3?=p��P  �C33�)�
C�s3                                    Bxs���  �          @�\)?�=q��녾�  �L��C��?�=q�aG�� ���	G�C���                                    Bxs�ސ  �          @�Q�?J=q���>���@��C�1�?J=q�tz��G���z�C��
                                    Bxs��6  �          @��R�#�
��녿���Xz�C���#�
�E�J�H�7
=C���                                    Bxs���  �          @�?xQ�����>u@J�HC�n?xQ��o\)�z��ᙚC�N                                    Bxs�
�  �          @��@Q���@+�B"��C��@Q��Vff?�ffAl��C�7
                                    Bxs�(  �          @���?�G�?��\@p��Bo�RB\)?�G��n{@w
=Bz�C�                                      Bxs�'�  �          @���?�{?��H@n{B_Q�B*p�?�{�
=q@���B�  C��                                    Bxs�6t  �          @�p�@z�?�33@|��Bf��Bff@z�k�@��\Bt  C��3                                    Bxs�E  �          @�(�?��?Ǯ@��Bv�B5�\?���Y��@�Q�B�8RC��{                                    Bxs�S�  �          @���?�
=@33@Mp�BV�Bp�?�
=<��
@s�
B��=?W
=                                    Bxs�bf  T          @|(�?\(�@ff@<��BF�
B��)?\(�>��@p  B��A�ff                                    Bxs�q  �          @�Q�?��H�У�@Mp�B^�C���?��H�E?���A��HC��=                                    Bxs��  �          @�
=>��z�H?��HA�ffC�O\>��}p������p��C�G�                                    Bxs��X  �          @��\�����U������C{\)���׿�
=�L���\�Co�q                                    Bxs���  �          @l(������AG��޸R��CzxR���ÿУ��E��f�HCl�)                                    Bxs���  �          @`  �xQ��;��У�����C{�R�xQ��{�<���f{Cn�f                                    Bxs��J  �          @z=q�\)�Z�H������
C�b��\)��
�HQ��^�\C~�3                                    Bxs���  �          @�z�8Q��z�H�z�H�]G�C��ÿ8Q��2�\�;��5��C�H                                    Bxs�ז  �          @�  ��(���33���
���CzQ쿼(��R�\�{�	z�Cv                                      Bxs��<  �          @�p����~{�\)���HC}����QG��\)��CzL�                                    Bxs���  �          @���У��p�׾\��\)Cv���У��>�R���	�
Cq\)                                    Bxs��  �          @�\)�#�
�����{��ffC��H�#�
�R�\��R��C���                                    Bxs�.  �          @�33>\��  �B�\�!��C��{>\�L(��:�H�){C�o\                                    Bxs� �  T          @��
?���y���(����C�+�?���>{�&ff�{C���                                    Bxs�/z  T          @�{�z�H�qG�?��A�
=Cuÿz�H��=q�!G����C�<)                                    Bxs�>   �          @����u���?   @�
=C�h��u�vff�����̏\C�@                                     Bxs�L�  �          @�
==���(��\��
=C��R=��Q��!��33C�\                                    Bxs�[l  �          @���>��
���Ϳp���L(�C�4{>��
�@  �AG��3�C��                                    Bxs�j  �          @�=q>��R�p  ������  C�\)>��R����N�R�T�RC��\                                    Bxs�x�  �          @[�=�Q쿔z��I��z�C�L�=�Q�?5�R�\� B���                                    Bxs��^  �          @��\>aG��   �c33�q�RC�!H>aG�>W
=��=q«z�B.�                                    Bxs��  �          @�  ��G�����[��t�C�S3��G�>�  �z=q¬\B���                                    Bxs���  �          @�논��
�!��7��B�RC��ͼ��
����q�¢\C�'�                                    Bxs��P  �          @�G��L���h�������
=C�k��L�Ϳ�
=�p  �z��C�f                                    Bxs���  �          @�����_\)�%�33C������Ǯ�����C{.                                    Bxs�М  �          @�
=�O\)�\���G���RC�j=�O\)�ٙ��qG��}�HCt��                                    Bxs��B  
�          @��H�#�
�z�H��{�ʣ�C���#�
�z��j�H�fC���                                    Bxs���  �          @��\�   �`���Q��33C��ÿ   �ٙ��x���C}�)                                    Bxs���  �          @�
=���qG������C���녿�
=�~�R�|��C}�3                                    Bxs�4  
Z          @�  ��  ��녿k��AG�C�^���  �I���E�1z�C��                                    Bxs��  �          @���=�Q���(��#�
��C��3=�Q��k��Q��Q�C��\                                    Bxs�(�  �          @�p�>W
=����L���ffC�1�>W
=��G��333�
�C�s3                                    Bxs�7&  �          @��
>Ǯ��33��p�����C�E>Ǯ�x���:=q��\C��3                                    Bxs�E�  �          @�Q�?
=q��녿���J=qC�+�?
=q�_\)�_\)�3  C�]q                                    Bxs�Tr  �          @�\)?.{��
=��ff�h��C��?.{�`  �p���:
=C�z�                                    Bxs�c  �          @�z�?J=q��  �fff��RC�E?J=q�|���dz��&Q�C���                                    Bxs�q�  �          @�{?Q������\(����C��)?Q��s33�[��%�\C�{                                    Bxs��d  �          @��\?p�����
��ff�5�C�7
?p���p���g��,33C�f                                    Bxs��
            @�p�?k���녿�����HC��?k�����Vff�z�C�AH                                    Bxs���  
�          @��?��H���\��33�mp�C�` ?��H����I����\C��\                                    Bxs��V  �          @���?W
=���\�1���HC��?W
=���H��G���C��{                                    Bxs���  �          @��?�z���ff��(����C��\?�z��K��p���@  C�                                    Bxs�ɢ  �          @��?�{��G�?\)@���C��?�{��\)�\)�ͮC���                                    Bxs��H  �          @��?����?�  AIC�(�?����Q����  C�b�                                    Bxs���  �          @�p�?��H����?�\)A`��C��{?��H���R����\)C��3                                    Bxs���  �          @��\?�p���?�(�A�p�C��?�p����H��
=�D  C���                                    Bxs�:  �          @�ff@z���(�@��AѮC���@z���\)�����^�RC�L�                                    Bxs��  �          @��R@��"�\@Y��B2ffC�5�@��~{?�G�A��C�/\                                    Bxs�!�  �          @���@��Vff?��A�G�C���@��u�W
=�/\)C�@                                     Bxs�0,  �          @���@�\�8��@
�HA�=qC�7
@�\�fff>�z�@z�HC�:�                                    Bxs�>�  �          @�(�@
=�3�
@�BQ�C���@
=�i��?   @���C�s3                                    Bxs�Mx  �          @�ff@���K�@��A�{C���@���u�=�G�?��C�=q                                    Bxs�\  
�          @�=q@(��`  ?��\A��HC�g�@(��h�ÿL���%�C���                                    Bxs�j�  �          @�(�@p��_\)?�G�A�p�C��{@p��y���\��p�C�#�                                    Bxs�yj  �          @�ff@ ��� ��@Dz�B#��C���@ ���p��?�G�Ax��C��q                                    Bxs��  �          @�z�@p��  @N�RB0\)C��@p��hQ�?��
A���C��                                    Bxs���  �          @��@�\�  @Z=qBAG�C��@�\�n�R?�
=A���C�U�                                    Bxs��\  �          @�=q?����|��@
=A�z�C��?�����
=�����w�C�AH                                    Bxs��  �          @�  @���\)?��Ac
=C��q@����R����j�RC��f                                    Bxs�¨  �          @��@)����  ?�ffA.�\C���@)����=q��33����C�f                                    Bxs��N  �          @�
=?�p���Q�?h��A
=C�O\?�p���{��\��  C��                                    Bxs���  �          @�p�@���33?��AIp�C��\@����R��\��C��                                    Bxs��  �          @���@ff���?��
AD(�C���@ff��Q����33C�                                    Bxs��@  �          @��H@�����H?���At  C�S3@�����
��  �c�C�C�                                    Bxs��  �          @�33@�R���?�33A{�
C���@�R���
��Q��Z=qC�h�                                    Bxs��  �          @�33@.�R���\?�Q�A��RC�"�@.�R���\�����"{C���                                    Bxs�)2  �          @��@1����
?���Ap��C�C�@1���p���z��W�C�'�                                    Bxs�7�  �          @��R@L(����>�Q�@`  C�P�@L(�����
=��
=C���                                    Bxs�F~  �          @��
@#33���R@(�A�ffC��f@#33������{C���                                    Bxs�U$  �          @��H@p��z�H@*=qA��HC�f@p���\)>#�
?�C���                                    Bxs�c�  �          @��@/\)�^{@:�HBC�!H@/\)��  ?�R@ָRC���                                    Bxs�rp  �          @�G�@,���O\)@K�B{C���@,����{?uA((�C��=                                    Bxs��  �          @���@4z��L��@E�BffC��@4z���33?fffA�C�w
                                    Bxs���  �          @���@AG��E�@G
=B
=C�>�@AG���Q�?}p�A+�C��                                    Bxs��b  �          @���@K��:�H@AG�B��C��)@K����\?�  A/
=C��)                                    Bxs��  �          @�=q@Vff�5@=p�B�C��q@Vff�~�R?}p�A,��C��                                    Bxs���  
0          @�\)@Vff�c�
?��A��C��H@Vff���׾�z��L(�C��                                    Bxs��T  
�          @�@:=q�P��@2�\Bz�C�޸@:=q��  ?!G�@�  C�33                                    Bxs���  
�          @��@���g
=@333B��C��)@������>�ff@��C�ٚ                                    Bxs��  �          @�(�@'��N�R@;�B��C��f@'�����?B�\A	p�C��3                                    Bxs��F  �          @���?��R����@,(�A��C�� ?��R����    <�C�<)                                    Bxs��  T          @��\?�G�����@A�B
p�C��?�G���Q�>�(�@��\C���                                    Bxs��  "          @��?����g�@[�B#�RC���?�����z�?z�HA,��C�\                                    Bxs�"8  T          @�@G��L(�@xQ�B7G�C�+�@G���Q�?˅A���C�z�                                    Bxs�0�  �          @��
?����u@��A��C�f?�����ff���˅C��
                                    Bxs�?�  �          @�  ��\)�l(��e��#��Cs(���\)��(�����)CU5�                                    Bxs�N*  "          @����
=q��
��\)�[(�C_�\�
=q?   ��.C&�H                                    Bxs�\�  
�          @���(��1��dz��0{Cd���(�����  �t�C>�{                                    Bxs�kv  
�          @�\)�p��Fff�Z�H�#33Cg���p��O\)��G��q=qCFT{                                    Bxs�z  
�          @�(��.�R�{�w
=�9\)C^:��.�R�L�����\�m  C5�                                    Bxs���  �          @��
�"�\�]p��E����Ci���"�\�����{�ep�CO�                                    Bxs��h  T          @�\)��(�����?L��B�Cx�þ�(���{=�Q�@mp�C|s3                                    Bxs��  
�          @���?��>��R@���B�G�A�?���p�@��B]33C���                                    Bxs���  
�          @�@33�޸R@�  Bv�HC�Ф@33��(�@Tz�B��C�33                                    Bxs��Z  �          @�33@���\@�p�Bv33C��\@�k�@`  B�C�@                                     